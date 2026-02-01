# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **PhD thesis research repository** for semantic communication and AI agent networks in 6G systems. It contains academic papers, technical documentation, and evolving research concepts rather than executable code.

**Core Research Question**: How should AI agents communicate in next-generation networks when traditional bit-transmission is replaced by semantic state synchronization?

**Key Innovation**: Token-Based Communication Protocol for Agent-Oriented 6G Networks - transmitting semantic state deltas instead of raw data.

**Target Venues**: IEEE INFOCOM, IEEE ICC, ACM SIGCOMM

## Repository Structure

The repository follows a paper-oriented organization matching PhD thesis chapters:

```
00-advisor-feedback/       # Professor guidance, feedback, and communication drafts
01-problem-formulation/    # Research questions, motivation, contributions
02-core-framework/         # Semantic State Communication (SSC) theory
03-technical-design/       # Protocol design and implementation details
04-background/             # Papers, related work, technical background
05-evaluation/             # Experiment design and scenarios
06-implementation/         # Implementation specifications
07-paper-drafts/           # LaTeX drafts and figures
08-code/                   # Simulation, prototype, evaluation code
09-project-logs/           # Phase completion reports, status snapshots, process docs
tools/                     # AI Agent tools and methodology (aider, OpenHands, SWE-agent)
archive/                   # Deprecated directions, evolution logs, and original source files
```

### Key Reference Files

**Start here for context**:
- `README.md` - Research overview and core contributions
- `ROADMAP.md` - Timeline, milestones, and phases

**Original research content** (preserved as references within structured directories):
- `02-core-framework/t3-original-reference.md` - Original SSC framework (canonical reference)
- `03-technical-design/t6-original-reference.md` - Original attention filtering design
- `01-problem-formulation/t8-core-arguments-reference.md` - Core arguments and defense strategy

### Directory-Specific Guidance

**00-advisor-feedback/**
- `professor-concepts-raw.md` - **Critical**: Contains advisor's core insights on token-based transmission
- `meeting-draft.md` - Prepared materials for advisor discussions
- `t8-advisor-email-v1.md` - Original advisor email draft with professor reply
- `t8-method-draft.md` - Method-focused communication draft
- `t8-updated-draft.md` - Updated version (Phase 2 progress)
- `t9-research-strategy.md` - Comprehensive research strategy analysis

**01-problem-formulation/**
- Research question definition, motivation, contributions
- `defense-strategy.md` - Distinguishes from ISAC/JSCC/MCP/traditional approaches
- `mathematical-system-model.md` - Information Bottleneck and Rate-Distortion formulation

**02-core-framework/**
- `semantic-state-sync.md` - SSC framework (refactored from t3.md)
- `semantic-token-definition.md` - Token as unit of communication
- `t3-original-reference.md` - Original version for reference

**03-technical-design/**
- `attention-filtering.md` - Source-side filtering using DeepSeek DSA (refactored from t6.md)
- `state-integration.md` - Receiver-side deterministic integration
- `t6-original-reference.md` - Original version for reference

**04-background/**
- `papers/` - Core reference PDFs
- `related-work/` - Comparative analysis with ISAC, JSCC, traditional communication
- `technical-background/` - Agent services, IoA architecture, DeepSeek details

**05-evaluation/**
- `scenarios.md` - Trace-driven simulation methodology

**06-implementation/**
- `ssc-pipeline-spec.md` - SSC protocol pipeline specification

**09-project-logs/**
- Phase completion reports (PHASE1/2/3), project status snapshots, restructure records
- `QUICK_START.md` - Immediate action guide for new work sessions

**tools/**
- `new_agent_idea.md` - AI Agent tools exploration (aider, OpenHands, SWE-agent + Ollama)

**archive/**
- `old-directions/` - Abandoned research paths (t1: O-RAN automation, t2: Edge RAG)
- `evolution-logs/` - Research progression records (t4, t5, t7)
- `original-sources/` - Pre-refactor originals (Architecture_Unification, Communication_Cost_Model, safari paradigm exploration)

## Core Technical Concepts

### The Paradigm Shift

**Traditional**: Source → Encoder → Channel → Decoder → Bits → Reconstruct
**This Research**: Shared World Model → Semantic State Δ → Tokenized Representation → State Synchronization

### Critical Terminology

- **SSC (Semantic State Communication)**: Synchronizing agent cognitive states, not transmitting data
- **Semantic Token**: Unit of communication (not bits/packets, but semantic state deltas)
- **Attention-Based Filtering**: Source decides "what's worth transmitting" based on task relevance
- **State Δ (Delta)**: Incremental change in semantic state
- **KV-Cache Sharing**: Transmitting transformer internal states between agents
- **DSA (DeepSeek Sparse Attention)**: Mechanism for efficient attention computation
- **Control/Data Plane Separation**: Control plane for task alignment, data plane for token transmission

### Theoretical Foundations

**Information Bottleneck Framework**:
```
min I(X; Z) - β I(Z; Y)
where Z = transmitted semantic state
```

**Rate-Distortion Objective**:
```
min R(S_t → Z_t) subject to D_task ≤ D_max
where D_task = task-oriented distortion
```

**Optimization Goal**:
```
max Task Success Rate
subject to: bandwidth, latency, QoS constraints
```

### Architecture Layers

**Internet of Agents (IoA) - 4 Layers**:
```
Application Scenarios
    ↓
Agent Coordination (Task orchestration)
    ↓
Agent Management (Discovery, capability notification)
    ↓
Infrastructure (Communication protocols, networking)
```

**FM-Powered Agent Services - 5 Layers**:
```
Application Layer (ChatGPT, Copilot)
    ↓
Agent Layer (Multi-agent, Planning, Memory, Tool Use)
    ↓
Model Layer (LLMs, Compression, Token Reduction)
    ↓
Resource Layer (Parallelism, Scaling)
    ↓
Execution Layer (Edge/Cloud, Hardware Optimization)
```

**SSC Protocol - Layer 5+**:
```
Semantic Protocol Layer
├── Control Plane (Task negotiation, model alignment)
└── Data Plane (Token transmission, flow control)
```

## Working Guidelines

### When Reading/Understanding the Research

1. **Follow the research progression**:
   - Start with `00-advisor-feedback/professor-concepts-raw.md` for core insights
   - Read `01-problem-formulation/research-question.md` for formal problem definition
   - Study `02-core-framework/semantic-state-sync.md` for the SSC framework
   - Review `03-technical-design/` for implementation mechanisms

2. **Check original sources when needed**:
   - `02-core-framework/t3-original-reference.md` - Original SSC framework thinking
   - `03-technical-design/t6-original-reference.md` - Original attention filtering thinking
   - `00-advisor-feedback/t8-advisor-email-v1.md` - Original advisor communication
   - Archive files explain why certain directions were abandoned

3. **Cross-reference extensively**:
   - These topics are deeply interconnected
   - Always verify consistency with `defense-strategy.md` when discussing differentiation
   - Check `mathematical-system-model.md` for formal definitions

### When Editing/Creating Documents

1. **Maintain technical precision**:
   - Use "semantic state synchronization" not "semantic communication"
   - Use "token-based transmission" not "token streaming"
   - Use "KV-cache sharing" when referring to transformer state transfer
   - Use "task success rate" not "accuracy" or "performance"

2. **Follow established notation**:
   - `S_t` for semantic state at time t
   - `Δ` (delta) for state changes
   - `Z_t` for transmitted representation
   - `L1/L2/L3` for network layers (from archived t1.md)

3. **Cite properly**:
   - Use bracketed numbers `[1]` for academic citations
   - Reference specific files with descriptive links
   - Maintain bilingual content (English technical + Traditional Chinese notes/feedback)

4. **Target top-tier quality**:
   - Writing targets IEEE INFOCOM, ICC, SIGCOMM standards
   - Include mathematical rigor where appropriate
   - Use precise system model descriptions
   - Provide clear architectural diagrams (as text descriptions)

### Critical Distinctions (Defense Strategy)

**What This Research Is NOT**:

1. **Not traditional Semantic Communication**:
   - Traditional: Transmit feature vectors instead of raw data (still data transmission)
   - **This work**: Transmit semantic state deltas (state synchronization)

2. **Not Agent Framework applications**:
   - LangChain/AutoGen: Assume infinite bandwidth, ignore communication costs
   - **This work**: Design communication protocols for bandwidth-constrained agent networks

3. **Not MCP application**:
   - MCP: Application-layer protocol for tool integration
   - **This work**: Lower-layer semantic protocol for state synchronization

4. **Not ISAC (Integrated Sensing and Communication)**:
   - ISAC: Transmit sensing data (external perception)
   - **This work**: Transmit cognitive state (internal reasoning)

5. **Not JSCC (Joint Source-Channel Coding)**:
   - JSCC: Optimize for data reconstruction
   - **This work**: Optimize for task success

**What This Research IS**:
- A fundamentally new communication paradigm for AI agents
- Token-based transmission with attention-driven filtering
- Task-oriented protocol design with control/data plane separation
- Communication system that optimizes for cognitive alignment, not bit accuracy

### Common Research Tasks

**Literature Review**:
```
Read papers in 04-background/papers/
→ Extract key concepts
→ Map to research directions in 01-problem-formulation/
→ Update related-work/ comparisons
→ Identify gaps
```

**Concept Refinement**:
```
Compare advisor feedback (00-advisor-feedback/)
→ Review current documents (01-03/)
→ Identify misalignments
→ Propose revisions
→ Verify consistency with defense-strategy.md
```

**Paper Writing**:
```
Problem statement (01-problem-formulation/)
→ Related work (04-background/related-work/)
→ Proposed framework (02-core-framework/)
→ Technical design (03-technical-design/)
→ Evaluation (05-evaluation/)
→ Assemble in 07-paper-drafts/
```

**Cross-Reference Checking**:
When editing any document, verify consistency with:
- Terminology in `02-core-framework/semantic-state-sync.md`
- Distinctions in `01-problem-formulation/defense-strategy.md`
- Constraints in `00-advisor-feedback/professor-concepts-raw.md`
- Architecture in `04-background/technical-background/`

### Writing Style

- **Language**: Mix of English (technical content) and Traditional Chinese (advisor feedback/notes)
- **Formalism**: High - use mathematical notation, system models, formal definitions
- **Citations**: Bracketed numbers `[1]`, maintain consistency
- **Diagrams**: Text-based descriptions with clear hierarchies

### Critical Pitfalls to Avoid

1. **Don't treat MCP as the solution** - Professor feedback explicitly states MCP is application-layer, insufficient for this research
2. **Don't focus on agent frameworks** - These ignore communication costs, which is our core concern
3. **Don't propose incremental improvements** to existing semantic communication - This research seeks a **paradigm shift**
4. **Don't separate KV-cache from communication protocol** - They're intrinsically linked
5. **Don't create new markdown files unnecessarily** - Use existing structure, edit existing files

## Current Status (as of 2026-01-24)

### Completed
- ✅ Repository restructured into paper-oriented organization
- ✅ Core framework documented (SSC, semantic tokens, attention filtering)
- ✅ Problem formulation with defense strategy
- ✅ Mathematical system model (IB + R-D)
- ✅ Technical design (source + receiver mechanisms)
- ✅ Evaluation methodology (trace-driven simulation)

### In Progress
- 🚧 Refining mathematical models
- 🚧 Literature review for related work section
- 🚧 Simulation environment design

### Next Steps (see ROADMAP.md)
- Phase 1 (Jan-Feb 2026): Problem Formulation finalization
- Phase 2 (Mar-Apr 2026): Framework Design refinement
- Phase 3 (Apr-May 2026): Technical Design completion
- Phase 4 (Jun-Jul 2026): Implementation & Simulation
- Phase 5 (Aug-Sep 2026): Evaluation & Analysis
- Phase 6 (Sep-Nov 2026): Paper Writing
- Phase 7 (Dec 2026+): Submission & Response

## Research Timeline Reference

**Target Submission**: IEEE INFOCOM 2027 (deadline ~Aug 2026) or IEEE ICC 2027 (deadline ~Oct 2026)

**Key Checkpoints** (see ROADMAP.md for details):
1. Feb 2026: Problem formulation approved by advisor
2. Apr 2026: Framework design finalized
3. May 2026: Technical design ready for implementation
4. Jul 2026: Initial experimental results
5. Nov 2026: Paper draft complete

## Support Files

- **ROADMAP.md** - Detailed timeline with phases, deliverables, and checkpoints
- **09-project-logs/QUICK_START.md** - Immediate action guide for new work sessions
- **09-project-logs/RESTRUCTURE_COMPLETE.md** - Documentation of recent reorganization

## Questions to Guide Work

When working on any task, ask:
1. **Does this align with the advisor's vision?** (Check `00-advisor-feedback/professor-concepts-raw.md`)
2. **Is the distinction from existing work clear?** (Check `defense-strategy.md`)
3. **Is the mathematical formulation rigorous?** (Check `mathematical-system-model.md`)
4. **Does this fit the paper structure?** (Check ROADMAP.md phases)
5. **Is the terminology consistent?** (Check `semantic-state-sync.md`)
