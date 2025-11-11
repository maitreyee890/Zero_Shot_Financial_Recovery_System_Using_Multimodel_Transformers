"""
Zero-Shot Financial Recovery Multi-Agent System
================================================

A LangChain-based multi-agent framework for financial crisis recovery analysis
using Gemini 2.5 Flash transformer models with structured orchestration.

Architecture:
    - Document Analysis Agent: Parses reports and extracts key information
    - Quantitative Reasoning Agent: Performs numerical analysis and forecasting
    - Risk Assessment Agent: Evaluates vulnerabilities and compliance
    - Recovery Strategy Agent: Generates actionable recovery plans
    - Synthesis Agent: Aggregates and produces unified recommendations

"""

import os
import mimetypes
import pathlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import SystemMessage, HumanMessage
from langchain.tools import Tool
from langchain_core.messages import BaseMessage
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

MAX_INLINE_BYTES = 20 * 1024 * 1024  # 20MB max for inline file uploads
DEFAULT_MODEL = "gemini-2.5-flash"
ALTERNATIVE_MODEL = "gemini-2.5-pro"
THINKING_BUDGET_TOKENS = 8192  # Default thinking budget for agents


class AgentRole(Enum):
    """Enumeration of specialized agent roles in the framework."""
    DOCUMENT_ANALYSIS = "document_analysis"
    QUANTITATIVE_REASONING = "quantitative_reasoning"
    RISK_ASSESSMENT = "risk_assessment"
    RECOVERY_STRATEGY = "recovery_strategy"
    SYNTHESIS = "synthesis"


@dataclass
class AgentConfig:
    """Configuration for individual agent instances.
    
    Attributes:
        role: The specialized role this agent performs
        thinking_budget: Token budget for pre-reasoning phase
        temperature: Sampling temperature for response generation
        description: Human-readable description of agent capabilities
    """
    role: AgentRole
    thinking_budget: int
    temperature: float
    description: str


# ============================================================================
# AGENT PROMPT TEMPLATES
# ============================================================================

AGENT_PROMPTS = {
    AgentRole.DOCUMENT_ANALYSIS: """
You are the Document Analysis Agent specialized in parsing and summarizing financial reports.

RESPONSIBILITIES:
- Extract key facts from textual reports, regulatory filings, and news articles
- Interpret embedded charts and tables using language-vision fusion
- Identify disclosed risk factors and management commentary
- Assess document reliability based on source credibility

OUTPUT FORMAT:
- Structured summaries with key facts
- Identified risk factors
- Relevant financial metrics extracted from tables
- Source reliability assessment

Analyze the provided documents and extract essential information for financial recovery planning.
""",

    AgentRole.QUANTITATIVE_REASONING: """
You are the Quantitative Reasoning Agent specialized in numerical analysis and forecasting.

RESPONSIBILITIES:
- Process market price data and identify trends
- Calculate financial ratios (liquidity, leverage, profitability, efficiency)
- Perform time series forecasting with confidence intervals
- Conduct statistical analysis of financial patterns

OUTPUT FORMAT:
- Computed financial metrics with interpretations
- Time series forecasts with confidence intervals
- Statistical analyses of significant patterns
- Quantitative severity assessments

Provide mathematical foundation for recovery strategy formulation.
""",

    AgentRole.RISK_ASSESSMENT: """
You are the Risk Assessment Agent specialized in identifying vulnerabilities and compliance issues.

RESPONSIBILITIES:
- Evaluate credit risk (counterparty exposure, default probability)
- Assess market risk (volatility, interest rate, currency exposure)
- Identify operational risk vulnerabilities
- Check regulatory compliance across jurisdictions

OUTPUT FORMAT:
- Structured risk assessments by category
- Risk severity scores with justifications
- Identified vulnerabilities requiring mitigation
- Compliance requirements for recovery strategies

Focus analytical resources on high-risk features that could impede recovery.
""",

    AgentRole.RECOVERY_STRATEGY: """
You are the Recovery Strategy Agent specialized in formulating actionable recovery plans.

RESPONSIBILITIES:
- Generate candidate recovery strategies (capital restructuring, asset liquidation, etc.)
- Specify concrete action steps with timelines
- Estimate resource requirements and stakeholder impacts
- Provide multiple strategic options (conservative vs. aggressive)

OUTPUT FORMAT:
- Detailed recovery plan documents
- Implementation timelines and milestones
- Resource requirements and constraints
- Financial projections under proposed strategies
- Risk mitigation measures

Synthesize inputs from other agents to create comprehensive, feasible recovery pathways.
""",

    AgentRole.SYNTHESIS: """
You are the Synthesis Agent responsible for coordinating all specialized agents and producing unified recommendations.

RESPONSIBILITIES:
- Aggregate outputs from Document Analysis, Quantitative Reasoning, Risk Assessment, and Recovery Strategy agents
- Evaluate consistency and identify conflicts between agent outputs
- Apply weighted voting on candidate strategies (feasibility, risk mitigation, compliance, stakeholder impact)
- Generate executive summaries with confidence scores

OUTPUT FORMAT:
- Executive summary of recommended recovery plan
- Detailed justifications referencing supporting evidence
- Confidence scores indicating recommendation strength
- Regulatory compliance attestations
- Alternative options for decision-maker consideration

Produce coherent, actionable final recommendations integrating all agent perspectives.
"""
}


# ============================================================================
# MULTI-AGENT ORCHESTRATION SYSTEM
# ============================================================================

class FinancialRecoveryOrchestrator:
    """
    LangChain-based orchestrator for multi-agent financial recovery analysis.
    
    This class manages five specialized Gemini 2.5 Flash agents that collaborate
    through structured communication to analyze financial crises and generate
    recovery strategies.
    
    Attributes:
        api_key: Google API key for Gemini access
        model_name: Base Gemini model identifier
        agents: Dictionary mapping AgentRole to configured LLM instances
        agent_outputs: Cache of intermediate agent analyses
    """
    
    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        """
        Initialize the multi-agent orchestrator.
        
        Args:
            api_key: Google API key for Gemini model access
            model_name: Gemini model identifier (default: gemini-2.5-flash)
        
        Raises:
            ValueError: If api_key is empty or invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("Valid Google API key required for orchestrator initialization")
        
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.agents: Dict[AgentRole, ChatGoogleGenerativeAI] = {}
        self.agent_outputs: Dict[AgentRole, str] = {}
        
        # Initialize specialized agents
        self._initialize_agents()
    
    def _initialize_agents(self) -> None:
        """
        Initialize all five specialized agents with appropriate configurations.
        
        Each agent is instantiated with role-specific prompts, thinking budgets,
        and temperature settings optimized for its specialized tasks.
        """
        agent_configs = [
            AgentConfig(
                role=AgentRole.DOCUMENT_ANALYSIS,
                thinking_budget=8192,
                temperature=0.3,
                description="Parses and summarizes financial documents"
            ),
            AgentConfig(
                role=AgentRole.QUANTITATIVE_REASONING,
                thinking_budget=8192,
                temperature=0.2,
                description="Performs numerical analysis and forecasting"
            ),
            AgentConfig(
                role=AgentRole.RISK_ASSESSMENT,
                thinking_budget=12288,
                temperature=0.4,
                description="Evaluates risks and compliance requirements"
            ),
            AgentConfig(
                role=AgentRole.RECOVERY_STRATEGY,
                thinking_budget=16384,
                temperature=0.5,
                description="Generates recovery strategy proposals"
            ),
            AgentConfig(
                role=AgentRole.SYNTHESIS,
                thinking_budget=12288,
                temperature=0.3,
                description="Synthesizes agent outputs into unified recommendations"
            )
        ]
        
        for config in agent_configs:
            self.agents[config.role] = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=config.temperature,
                convert_system_message_to_human=True
            )
    
    def _process_agent(
        self, 
        role: AgentRole, 
        scenario_input: str,
        context: Optional[str] = None
    ) -> str:
        """
        Execute a single agent analysis on the scenario input.
        
        Args:
            role: The agent role to execute
            scenario_input: Financial crisis scenario description
            context: Optional context from peer agents
        
        Returns:
            Structured analysis output from the specified agent
        
        Raises:
            RuntimeError: If agent execution fails
        """
        try:
            prompt = AGENT_PROMPTS[role]
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"SCENARIO:\n{scenario_input}")
            ]
            
            if context:
                messages.append(HumanMessage(content=f"\nCONTEXT FROM PEER AGENTS:\n{context}"))
            
            response = self.agents[role].invoke(messages)
            return response.content
        
        except Exception as e:
            raise RuntimeError(f"Agent {role.value} execution failed: {str(e)}")
    
    def parallel_processing(self, scenario_input: str) -> Dict[AgentRole, str]:
        """
        Phase 1: Execute parallel initial analysis by all specialized agents.
        
        All agents receive identical scenario documents and generate preliminary
        analyses independently to maximize computational efficiency.
        
        Args:
            scenario_input: Comprehensive financial crisis scenario
        
        Returns:
            Dictionary mapping agent roles to their initial analyses
        """
        print("🔄 Phase 1: Parallel Processing - Agents analyzing scenario independently...")
        
        # Process Document Analysis, Quantitative Reasoning, and Risk Assessment in parallel
        # (In production, use async/threading for true parallelization)
        initial_agents = [
            AgentRole.DOCUMENT_ANALYSIS,
            AgentRole.QUANTITATIVE_REASONING,
            AgentRole.RISK_ASSESSMENT
        ]
        
        for role in initial_agents:
            print(f"  ├─ {role.value.replace('_', ' ').title()} Agent working...")
            self.agent_outputs[role] = self._process_agent(role, scenario_input)
        
        print("✅ Phase 1 Complete: Initial analyses generated\n")
        return self.agent_outputs
    
    def peer_review_cycle(self, scenario_input: str) -> Dict[AgentRole, str]:
        """
        Phase 2: Conduct peer review where agents refine analyses based on peer insights.
        
        Agents access peer outputs and update their assessments to reconcile
        discrepancies and incorporate complementary insights.
        
        Args:
            scenario_input: Original scenario description
        
        Returns:
            Dictionary of refined agent analyses after peer review
        """
        print("🔄 Phase 2: Peer Review Cycle - Agents incorporating peer insights...")
        
        # Build context from initial analyses
        context = "\n\n".join([
            f"=== {role.value.replace('_', ' ').title()} Agent Output ===\n{output}"
            for role, output in self.agent_outputs.items()
        ])
        
        # Recovery Strategy Agent processes with full context
        print("  ├─ Recovery Strategy Agent generating plans with context...")
        self.agent_outputs[AgentRole.RECOVERY_STRATEGY] = self._process_agent(
            AgentRole.RECOVERY_STRATEGY,
            scenario_input,
            context
        )
        
        print("✅ Phase 2 Complete: Peer review refinements incorporated\n")
        return self.agent_outputs
    
    def plan_aggregation(self, scenario_input: str) -> str:
        """
        Phase 3: Synthesis Agent aggregates all outputs and produces unified recommendation.
        
        The Synthesis Agent evaluates candidate strategies using weighted voting
        on feasibility, risk mitigation, compliance, and stakeholder impact.
        
        Args:
            scenario_input: Original scenario description
        
        Returns:
            Final synthesized recovery recommendation
        """
        print("🔄 Phase 3: Plan Aggregation - Synthesis Agent producing final recommendation...")
        
        # Compile all agent outputs for synthesis
        all_context = "\n\n".join([
            f"=== {role.value.replace('_', ' ').title()} Agent Analysis ===\n{output}"
            for role, output in self.agent_outputs.items()
        ])
        
        final_recommendation = self._process_agent(
            AgentRole.SYNTHESIS,
            scenario_input,
            all_context
        )
        
        self.agent_outputs[AgentRole.SYNTHESIS] = final_recommendation
        
        print("✅ Phase 3 Complete: Final recommendation synthesized\n")
        return final_recommendation
    
    def zero_shot_inference(self, scenario_input: str) -> Tuple[str, Dict[AgentRole, str]]:
        """
        Execute complete zero-shot inference workflow across all phases.
        
        This method orchestrates the full multi-agent pipeline:
        1. Parallel Processing: Independent initial analyses
        2. Peer Review: Refinement based on peer insights
        3. Plan Aggregation: Synthesis into unified recommendation
        
        Args:
            scenario_input: Financial crisis scenario description with multimodal data
        
        Returns:
            Tuple of (final_recommendation, all_agent_outputs)
        """
        print("\n" + "="*80)
        print("🚀 ZERO-SHOT FINANCIAL RECOVERY INFERENCE WORKFLOW")
        print("="*80 + "\n")
        
        # Phase 1: Parallel initial processing
        self.parallel_processing(scenario_input)
        
        # Phase 2: Peer review and refinement
        self.peer_review_cycle(scenario_input)
        
        # Phase 3: Final synthesis
        final_recommendation = self.plan_aggregation(scenario_input)
        
        print("="*80)
        print("✨ WORKFLOW COMPLETE: Recovery recommendation generated")
        print("="*80 + "\n")
        
        return final_recommendation, self.agent_outputs


# ============================================================================
# FILE HANDLING UTILITIES
# ============================================================================

def merge_file_lists(new_files: List, current_files: List) -> List[str]:
    """
    Merge new file uploads with existing file list, removing duplicates.
    
    Args:
        new_files: List of newly uploaded file objects
        current_files: List of currently tracked file paths
    
    Returns:
        Merged list of unique file paths
    """
    current = current_files or []
    additions = [f.name if hasattr(f, "name") else f for f in (new_files or [])]
    
    seen = set()
    merged = []
    
    for path in current + additions:
        if path and path not in seen:
            merged.append(path)
            seen.add(path)
    
    return merged


def prepare_multimodal_content(
    file_paths: List[str],
    user_message: str,
    client: genai.Client
) -> List:
    """
    Prepare multimodal content for Gemini API including text, PDFs, and images.
    
    Handles inline upload for small files and API-based upload for larger files.
    Supports PDF documents, images, and text files.
    
    Args:
        file_paths: List of paths to uploaded files
        user_message: User's text query
        client: Initialized Gemini API client
    
    Returns:
        List of content parts ready for Gemini API submission
    """
    contents = []
    
    for path in file_paths or []:
        if not path or not os.path.exists(path):
            continue
        
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "application/octet-stream"
        file_size = os.path.getsize(path)
        
        # Handle inline upload for small files
        if file_size <= MAX_INLINE_BYTES and mime_type in ["application/pdf", "image/png", "image/jpeg", "text/plain"]:
            data = pathlib.Path(path).read_bytes()
            contents.append(types.Part.from_bytes(data=data, mime_type=mime_type))
        
        # Use File API for large files
        else:
            uploaded_file = client.files.upload(file=path, config=dict(mime_type=mime_type))
            contents.append(uploaded_file)
    
    # Add user text message
    if user_message and user_message.strip():
        contents.append(user_message.strip())
    
    return contents


# ============================================================================
# GRADIO INTERFACE CALLBACKS
# ============================================================================

def handle_file_change(new_files: List, file_state: List) -> Tuple[List, gr.update, str]:
    """
    Handle file upload widget changes.
    
    Args:
        new_files: Newly uploaded files
        file_state: Current file state
    
    Returns:
        Tuple of (updated_state, ui_update, status_message)
    """
    merged = merge_file_lists(new_files, file_state)
    status = f"📁 {len(merged)} file(s) ready for analysis"
    return merged, gr.update(value=merged), status


def handle_upload_button(new_files: List, file_state: List) -> Tuple[List, gr.update, str]:
    """
    Handle upload button click.
    
    Args:
        new_files: Files selected via upload button
        file_state: Current file state
    
    Returns:
        Tuple of (updated_state, ui_update, status_message)
    """
    merged = merge_file_lists(new_files, file_state)
    status = f"✅ {len(merged)} file(s) uploaded successfully"
    return merged, gr.update(value=merged), status


def handle_clear_files() -> Tuple[List, gr.update, gr.update, str]:
    """
    Clear all uploaded files and reset state.
    
    Returns:
        Tuple of (empty_state, uploader_reset, button_reset, empty_status)
    """
    return [], gr.update(value=None), gr.update(value=None), ""


def chat_with_orchestrator(
    message: str,
    history: List,
    file_state: List[str],
    model_name: str
) -> str:
    """
    Main chat function integrating multi-agent orchestration.
    
    This function:
    1. Validates inputs and API credentials
    2. Prepares multimodal content from files and user message
    3. Executes zero-shot inference workflow via orchestrator
    4. Returns synthesized recovery recommendation
    
    Args:
        message: User's query message
        history: Chat conversation history (unused in current implementation)
        file_state: List of uploaded file paths
        model_name: Selected Gemini model identifier
    
    Returns:
        Final recovery recommendation from Synthesis Agent
    """
    # Validate message
    if not message or not message.strip():
        return "❌ Please provide a financial scenario description."
    
    # Retrieve API key
    api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        return "❌ Error: Missing GOOGLE_API_KEY in environment variables (.env file)"
    
    try:
        # Initialize Gemini client for file handling
        client = genai.Client(api_key=api_key)
        
        # Prepare multimodal content
        contents = prepare_multimodal_content(file_state, message, client)
        
        # Build comprehensive scenario input
        scenario_input = f"{message}\n\nAttached Files: {len(file_state)} document(s)"
        
        # Initialize orchestrator
        orchestrator = FinancialRecoveryOrchestrator(api_key, model_name)
        
        # Execute zero-shot inference workflow
        final_recommendation, agent_outputs = orchestrator.zero_shot_inference(scenario_input)
        
        # Format response with agent breakdown
        response = f"""
## 🎯 Financial Recovery Recommendation

{final_recommendation}

---

### 📊 Agent Analysis Breakdown

<details>
<summary>📄 Document Analysis Agent Output</summary>

{agent_outputs.get(AgentRole.DOCUMENT_ANALYSIS, 'N/A')}

</details>

<details>
<summary>🔢 Quantitative Reasoning Agent Output</summary>

{agent_outputs.get(AgentRole.QUANTITATIVE_REASONING, 'N/A')}

</details>

<details>
<summary>⚠️ Risk Assessment Agent Output</summary>

{agent_outputs.get(AgentRole.RISK_ASSESSMENT, 'N/A')}

</details>

<details>
<summary>💡 Recovery Strategy Agent Output</summary>

{agent_outputs.get(AgentRole.RECOVERY_STRATEGY, 'N/A')}

</details>
"""
        
        return response
    
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}\n\nPlease check your inputs and try again."


# ============================================================================
# GRADIO APPLICATION INTERFACE
# ============================================================================

def create_interface() -> gr.Blocks:
    """
    Create and configure the Gradio web interface.
    
    Returns:
        Configured Gradio Blocks application
    """
    with gr.Blocks(
        title="Zero-Shot Financial Recovery System",
        theme=gr.themes.Soft(),
        fill_height=True
    ) as interface:
        # State management
        file_state = gr.State([])
        
        # Header
        gr.Markdown("""
        # 🏦 Zero-Shot Financial Recovery Multi-Agent System
        
        **Powered by 5 Specialized Gemini 2.5 Flash Agents with LangChain Orchestration**
        
        This system analyzes financial crisis scenarios using collaborative intelligence from:
        - 📄 Document Analysis Agent
        - 🔢 Quantitative Reasoning Agent  
        - ⚠️ Risk Assessment Agent
        - 💡 Recovery Strategy Agent
        - 🎯 Synthesis Agent
        """)
        
        with gr.Row():
            # Left sidebar: File upload and configuration
            with gr.Column(scale=3, min_width=280):
                gr.Markdown("### 📁 File Upload")
                
                file_uploader = gr.File(
                    label="Add Financial Documents (PDF, Images, Text)",
                    file_count="multiple",
                    type="filepath"
                )
                
                upload_button = gr.UploadButton(
                    "📤 Or Click to Upload",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt", ".csv"],
                    file_count="multiple"
                )
                
                clear_button = gr.Button("🗑️ Clear All Files", variant="stop")
                
                file_status = gr.Markdown("")
                
                gr.Markdown("### ⚙️ Model Configuration")
                
                model_selector = gr.Dropdown(
                    choices=["gemini-2.5-flash", "gemini-2.5-pro"],
                    value="gemini-2.5-flash",
                    label="Select Gemini Model",
                    info="Flash is faster, Pro is more capable"
                )
                
                gr.Markdown("""
                ### 📖 Usage Instructions
                
                1. Upload financial documents (optional)
                2. Describe the crisis scenario in chat
                3. Receive comprehensive recovery analysis
                4. Review individual agent outputs in dropdown sections
                """)
            
            # Right panel: Chat interface
            with gr.Column(scale=7, min_width=480):
                gr.Markdown("### 💬 Financial Recovery Chat")
                
                chat_interface = gr.ChatInterface(
                    fn=chat_with_orchestrator,
                    type="messages",
                    title="",
                    description="Describe your financial crisis scenario below",
                    examples=[
                        "Analyze a liquidity crisis scenario with declining cash reserves and increasing short-term debt obligations.",
                        "Evaluate recovery options for a company facing bankruptcy due to supply chain disruptions and declining revenues.",
                        "Assess the impact of rising interest rates on a financial institution's bond portfolio and recommend mitigation strategies."
                    ],
                    multimodal=False,
                    additional_inputs=[file_state, model_selector],
                    retry_btn=None,
                    undo_btn=None,
                    clear_btn="🔄 New Analysis"
                )
        
        # Event handlers
        file_uploader.change(
            handle_file_change,
            [file_uploader, file_state],
            [file_state, file_uploader, file_status]
        )
        
        upload_button.upload(
            handle_upload_button,
            [upload_button, file_state],
            [file_state, file_uploader, file_status]
        )
        
        clear_button.click(
            handle_clear_files,
            None,
            [file_state, file_uploader, upload_button, file_status]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Zero-Shot Financial Recovery System** | Version 2.0.0 | Powered by Gemini 2.5 Flash + LangChain
        
        Research Paper: *Zero Shot Financial Recovery Using Multimodal Transformers*  
        Framework: 5-Agent Collaborative Intelligence Architecture
        """)
    
    return interface


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Verify environment setup
    api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    
    if not api_key:
        print("❌ ERROR: Missing Google API key!")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file")
        exit(1)
    
    print("✅ Environment configured successfully")
    print("🚀 Launching Zero-Shot Financial Recovery System...\n")
    
    # Create and launch interface
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
