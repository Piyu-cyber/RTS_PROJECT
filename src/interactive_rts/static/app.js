/**
 * Academic Implementation Walkthrough Controller
 * Project: Deadline-Aware Adaptive Inference for Real-Time Vision Systems
 * 
 * Manages:
 * 1. 10-Step Sequential Walkthrough State Machine.
 * 2. Visual highlighting of diagram blocks, connectors, and decision diamonds.
 * 3. Rich, technical explanation panel rendering with worked examples and file mappings.
 * 4. Keyboard navigation (Left/Right arrows) and step pill tab synchronization.
 */

document.addEventListener('DOMContentLoaded', () => {
  // =========================================================================
  // WALKTHROUGH STEPS DATA REGISTRY
  // =========================================================================
  const STEPS_DATA = [
    {
      step: 1,
      title: "Step 1: Image Input & Sensory Intake",
      activeNodeId: "node-step-1",
      activeBranchId: null,
      whatHappens: "The real-time vision system receives an image frame from an external source—such as an autonomous vehicle camera or dataset iterator (e.g. BDD100K test split).",
      whyNecessary: "In real-time systems, an raw image is purely pixel data $(3 \times 224 \times 224)$. Before it can be processed by a real-time scheduler, it must be ingested into the input buffer and paired with sensor acquisition metadata.",
      inputs: [
        { label: "Raw Data", val: "RGB Image Tensor (3 × 224 × 224 float32)" },
        { label: "Acquisition Timestamp", val: "Wall-clock sensor capture time (ms)" },
        { label: "Source", val: "Autonomous Vehicle Camera / data/bdd100k/" }
      ],
      outputs: [
        { label: "Buffer Reference", val: "Raw frame memory pointer in input buffer" },
        { label: "Ground Truth (optional)", val: "BDD100K driving scene class (e.g. 'city street', 'highway')" }
      ],
      sourceFile: "src/interactive_rts/workload_manager.py",
      sourceMethod: "WorkloadManager.validate_and_save_upload()",
      workedExample: "Sample frame ingested: c2461a5a-936bf379.jpg (224 × 224 × 3 RGB). Ground truth label: 'residential' (class index 2). Buffer allocated at t = 0.0 ms.",
      nextStepSummary: "Wrap the ingested frame into a formal Real-Time Request object with arrival and deadline timestamps."
    },
    {
      step: 2,
      title: "Step 2: Request Creation & Timing Parameters",
      activeNodeId: "node-step-2",
      activeBranchId: null,
      whatHappens: "The intake layer encapsulates the raw image frame into an instance of the Request dataclass. It assigns an arrival timestamp $r_i$ and a relative deadline $D_i$, then computes the absolute deadline: $d_i = r_i + D_i$.",
      whyNecessary: "A real-time scheduler cannot reason about pixel arrays; it requires explicit temporal parameters ($r_i, D_i, d_i$). Establishing $d_i$ defines the strict wall-clock boundary beyond which the inference result loses utility.",
      inputs: [
        { label: "Frame Pointer", val: "Image reference from Step 1" },
        { label: "Arrival Time ($r_i$)", val: "10.0 ms (illustrative example)" },
        { label: "Relative Deadline ($D_i$)", val: "5.0 ms (illustrative example)" }
      ],
      outputs: [
        { label: "Request Object", val: "Request(request_id='R01', r_i=10.0, D_i=5.0, d_i=15.0)" },
        { label: "Initial State", val: "RequestState.ARRIVED" }
      ],
      sourceFile: "src/phase24/request.py",
      sourceMethod: "Request.__post_init__() [lines 50–75]",
      workedExample: "Formula: d_i = r_i + D_i\nGiven: r_i = 10.0 ms, D_i = 5.0 ms\nResult: Absolute deadline d_i = 10.0 + 5.0 = 15.0 ms.\n(Note: Illustrative values to demonstrate timing definitions).",
      nextStepSummary: "Evaluate whether the newly created request can be admitted without violating timing constraints."
    },
    {
      step: 3,
      title: "Step 3: Admission Control Feasibility Check",
      activeNodeId: "node-step-3",
      activeBranchId: "branch-node-rej",
      whatHappens: "Before a request enters the queue, the Admission Controller tests whether it can complete on time given the ongoing GPU execution and current queue backlog. If predicted to miss its deadline, the request is REJECTED immediately at arrival.",
      whyNecessary: "Executing doomed requests wastes GPU cycles and causes cascading delays for subsequent requests. By dropping infeasible requests at arrival, the system protects already-admitted requests, preventing catastrophic queue collapse.",
      inputs: [
        { label: "Current Time ($t$)", val: "10.0 ms" },
        { label: "GPU Busy Until", val: "12.0 ms (ongoing execution)" },
        { label: "Queue Backlog", val: "∑ C(q) for all q ∈ Q (e.g. 3.5 ms)" },
        { label: "Estimated Latency", val: "C_est = 2.71 ms (EViT-0.5 fastest profile)" }
      ],
      outputs: [
        { label: "Decision", val: "Boolean (True = Enqueue, False = Reject)" },
        { label: "Rejection Terminal", val: "If False: RequestOutcome.REJECTED (Branch B)" }
      ],
      sourceFile: "src/phase24/admission_control.py",
      sourceMethod: "AdmissionController.is_feasible() [lines 38–70]",
      workedExample: "Rule: t_est_finish = max(t, t_busy) + backlog + C_est <= d_i\nCalculation: max(10.0, 12.0) + 3.5 + 2.71 = 18.21 ms.\nCandidate Deadline: d_i = 15.0 ms.\nComparison: 18.21 ms > 15.0 ms => Infeasible => REJECTED at t = 10.0 ms.",
      nextStepSummary: "If admitted, place the request into the Priority Queue awaiting GPU execution."
    },
    {
      step: 4,
      title: "Step 4: Priority Queue Buffering",
      activeNodeId: "node-step-4",
      activeBranchId: null,
      whatHappens: "Admitted requests buffer in a priority queue while the single-lane GPU finishes earlier work. The request accumulates queue waiting time: $W_i = t_{\\text{dispatch}} - r_i$.",
      whyNecessary: "Because batch-1 GPU inference is non-preemptive and serial, newly arriving tasks must be queued. Waiting directly consumes available deadline budget: remaining slack shrinks by $W_i$ before execution starts.",
      inputs: [
        { label: "Admitted Request", val: "Request object with state RequestState.QUEUED" },
        { label: "Queue Structure", val: "Binary heap / list ordered by priority comparator" }
      ],
      outputs: [
        { label: "Queue Sequence", val: "Ordered queue state Q = [R_head, R_2, ...]" },
        { label: "Queue Length", val: "Current queue backlog count" }
      ],
      sourceFile: "src/phase24/priority_queue.py",
      sourceMethod: "PriorityQueue.push() / list_requests() [lines 15–45]",
      workedExample: "Request arrives at r_i = 2.0 ms with d_i = 10.0 ms.\nGPU is busy until t = 5.0 ms.\nWaiting time: W_i = 5.0 - 2.0 = 3.0 ms.\nRemaining time budget decreases from 8.0 ms to 5.0 ms.",
      nextStepSummary: "When the GPU becomes idle, the scheduler dispatches the highest-priority request."
    },
    {
      step: 5,
      title: "Step 5: Scheduling Policy Dispatch (FIFO vs. EDF)",
      activeNodeId: "node-step-5",
      activeBranchId: null,
      whatHappens: "When the GPU completes an inference, the Scheduler pops the next request from the queue:\n• FIFO (First-In, First-Out): Dispatches earliest arrival min(r_i).\n• EDF (Earliest Deadline First): Dispatches earliest absolute deadline min(d_i).",
      whyNecessary: "FIFO is deadline-agnostic: a request with a loose deadline can block an urgent request, causing it to miss its deadline. EDF prioritizes urgency to minimize maximum lateness.",
      inputs: [
        { label: "Queue Buffer", val: "[R_loose (r=0.5, d=10.5), R_urgent (r=1.0, d=7.5)]" },
        { label: "Policy Selection", val: "FIFO vs. EDF configuration switch" }
      ],
      outputs: [
        { label: "Dispatched Request", val: "Selected candidate request R*" },
        { label: "Dispatch Time", val: "t_dispatch = current_time_ms" }
      ],
      sourceFile: "src/phase24/priority_queue.py",
      sourceMethod: "FIFOQueue.pop() vs. EDFQueue.pop() [lines 50–90]",
      workedExample: "Workload demonstrating EDF superiority:\nR_loose: arr=0.5 ms, dead=10.5 ms; R_urgent: arr=1.0 ms, dead=7.5 ms.\n• Under FIFO: Pops R_loose first. R_urgent waits and finishes at 10.5 ms > 7.5 ms (LATE!).\n• Under EDF: Pops R_urgent first (7.5 < 10.5). Finishes at 7.0 ms <= 7.5 ms (ON-TIME!). Both meet deadlines under EDF.",
      nextStepSummary: "Pass the dispatched request to the Adaptive Controller to pick the best model configuration."
    },
    {
      step: 6,
      title: "Step 6: Adaptive Model Controller (Token Pruning)",
      activeNodeId: "node-step-6",
      activeBranchId: null,
      whatHappens: "At the moment of dispatch, the Adaptive Controller computes the remaining slack for each candidate model:\n$s_i(M) = d_i - (t_{\\text{dispatch}} + C_M)$. It selects the highest-accuracy model whose estimated latency fits within the deadline ($s_i \\ge 0$) and meets accuracy threshold $\\alpha \\ge 92.0\\%$.",
      whyNecessary: "Fixed models cannot adjust execution time. When queue delays leave less than 3.5 ms of slack, DeiT-Small (3.78 ms) will miss the deadline. EViT can prune tokens down to 49 tokens (2.71 ms), saving up to 1.07 ms and completing on time.",
      inputs: [
        { label: "Dispatch Slack Budget", val: "s_i = d_i - t_dispatch (e.g. 3.0 ms)" },
        { label: "Accuracy Threshold α", val: "92.0% validation accuracy lower bound" },
        { label: "Model Profiles", val: "DeiT (3.78ms), EViT-0.9 (3.51ms), EViT-0.7 (3.25ms), EViT-0.5 (2.71ms)" }
      ],
      outputs: [
        { label: "Selected Model", val: "Optimal config_id (e.g. 'evit_small_k05_fuse')" },
        { label: "Estimated Latency", val: "C_est = 2.710 ms" }
      ],
      sourceFile: "src/phase24/adaptive_controller.py",
      sourceMethod: "AdaptiveController.select_model() [lines 35–85]",
      workedExample: "Dispatch at t = 4.0 ms, Deadline d_i = 7.0 ms. Available budget = 3.0 ms.\n• DeiT-Small: 3.78 ms => Slack = 3.0 - 3.78 = -0.78 ms (MISSED)\n• EViT-0.9: 3.51 ms => Slack = 3.0 - 3.51 = -0.51 ms (MISSED)\n• EViT-0.5: 2.71 ms => Slack = 3.0 - 2.71 = +0.29 ms (FEASIBLE!)\nController selects EViT-Small 0.5 (49 tokens, 91.44% acc).",
      nextStepSummary: "Forward the selected model and image to the Vision Inference Engine for execution."
    },
    {
      step: 7,
      title: "Step 7: Vision Model Execution & GPU Inference",
      activeNodeId: "node-step-7",
      activeBranchId: null,
      whatHappens: "The selected Vision Transformer executes batch-1 forward inference on the NVIDIA RTX 4060 GPU:\n• Mode A (Discrete Simulation): Uses profiled empirical p95 latency from model_profiles.json.\n• Mode B (Physical GPU): Replays pre-captured CUDA Graphs to eliminate CPU kernel launch overhead, achieving sub-4ms physical execution.",
      whyNecessary: "Standard eager PyTorch has 13–15 ms driver launch overhead on batch 1, which exceeds tight real-time deadlines. CUDA Graph replay captures the entire computation DAG into an executable GPU graph, unlocking true 2.7–3.8 ms inference.",
      inputs: [
        { label: "Selected Model", val: "Configured EViT / DeiT model instance" },
        { label: "Execution Mode", val: "Simulation (Mode A) or RTX 4060 GPU (Mode B)" },
        { label: "Input Tensor", val: "1 × 3 × 224 × 224 normalized float32 tensor" }
      ],
      outputs: [
        { label: "Execution Time (C_i)", val: "Measured wall-clock latency (e.g. 2.72 ms)" },
        { label: "Logits / Prediction", val: "Top-1 class prediction & Softmax confidence" }
      ],
      sourceFile: "src/phase24/gpu_trace_replay.py / src/interactive_rts/gpu_inference.py",
      sourceMethod: "CUDAGraphRunner.replay() / run_inference()",
      workedExample: "Physical RTX 4060 forward pass via CUDA Graph:\nModel: EViT-Small 0.5 (49 patch tokens + 1 fused token + 1 CLS token).\nPhysical latency: 2.72 ms.\nPrediction: 'city street' (class ID 0, confidence 94.2%).",
      nextStepSummary: "Compare the actual completion timestamp against the absolute deadline."
    },
    {
      step: 8,
      title: "Step 8: Deadline Monitor & Timing Evaluation",
      activeNodeId: "node-step-8",
      activeBranchId: null,
      whatHappens: "The Deadline Monitor evaluates whether execution finished before the absolute deadline:\nCompletion timestamp: $C_i = t_{\\text{dispatch}} + C_{\\text{actual}}$.\n• If $C_i \\le d_i$: Marked ON-TIME (Success).\n• If $C_i > d_i$: Marked LATE (Soft real-time miss).",
      whyNecessary: "This evaluates whether the system met its Quality-of-Service (QoS) guarantee. In real-time systems, an accurate classification is considered a failure if delivered after the deadline.",
      inputs: [
        { label: "Completion Time ($C_i$)", val: "t_dispatch + C_actual (e.g. 4.0 + 2.72 = 6.72 ms)" },
        { label: "Absolute Deadline ($d_i$)", val: "7.0 ms" }
      ],
      outputs: [
        { label: "Outcome State", val: "RequestOutcome.ON_TIME or RequestOutcome.LATE" },
        { label: "Final Slack", val: "d_i - C_i = +0.28 ms" }
      ],
      sourceFile: "src/phase24/request.py / src/interactive_rts/engine.py",
      sourceMethod: "Request.mark_completed() [lines 80–105]",
      workedExample: "Completion check:\nC_i = 6.72 ms, d_i = 7.0 ms.\nCondition: 6.72 ms <= 7.0 ms => TRUE.\nOutcome: ON_TIME (Slack: +0.28 ms).\n(Had DeiT-Small run: C_i = 4.0 + 3.78 = 7.78 ms > 7.0 ms => LATE by 0.78 ms).",
      nextStepSummary: "Partition the request into its terminal RTS state and verify system-wide conservation laws."
    },
    {
      step: 9,
      title: "Step 9: Final Request Outcome & RTS Accounting",
      activeNodeId: "node-step-9",
      activeBranchId: "branch-node-ontime",
      whatHappens: "The system logs the request into its final terminal state and tallies system-wide metrics. Every request offered to the system belongs to exactly one of four mutually exclusive terminal categories.",
      whyNecessary: "Formal real-time systems accounting requires strict conservation verification. Late tasks must not be hidden or conflated with rejected tasks.",
      inputs: [
        { label: "Terminal Sets", val: "{On-Time, Late, Rejected, Expired}" }
      ],
      outputs: [
        { label: "On-Time Ratio", val: "N_ontime / N_offered (Primary metric)" },
        { label: "Admission Ratio", val: "N_admitted / N_offered" },
        { label: "Mean Turnaround", val: "Average (C_i - r_i) across completed tasks" }
      ],
      sourceFile: "src/phase24/discrete_event_simulator.py",
      sourceMethod: "DiscreteEventSimulator._verify_accounting() [lines 310–350]",
      workedExample: "RTS Conservation Identities Verification:\n1. N_adm + N_rej = N_off (76 + 24 = 100)\n2. N_comp + N_exp = N_adm (76 + 0 = 76)\n3. N_ontime + N_late = N_comp (76 + 0 = 76)\nResult: All 3 conservation laws verified mathematically.",
      nextStepSummary: "Synthesize the entire architecture and review the overall experimental performance."
    },
    {
      step: 10,
      title: "Step 10: Complete Proposed Architecture (Holistic View)",
      activeNodeId: "ALL",
      activeBranchId: null,
      whatHappens: "The entire real-time pipeline operates end-to-end: Dynamic Input → Request Creation → Admission Control → Priority Queue → Scheduler Dispatch → Adaptive Controller → GPU Inference → Deadline Monitor → Final Output.",
      whyNecessary: "Shows how the three mechanisms work together: Admission Control prevents catastrophic backlog explosion; EDF prioritizes urgency; Adaptive Token Pruning recovers crucial latency slack. Together, they achieve a 70 percentage-point improvement in on-time completion under overload.",
      inputs: [
        { label: "Workload Stream", val: "Continuous image request stream" },
        { label: "Full Architecture", val: "EDF + Admission Control + Adaptive EViT" }
      ],
      outputs: [
        { label: "On-Time Completion", val: "76.0% (vs. 6.0% for baseline on periodic_high_tight)" },
        { label: "Accuracy Preserved", val: "92.48% test accuracy (within 0.19% of baseline)" }
      ],
      sourceFile: "src/interactive_rts/engine.py",
      sourceMethod: "InteractiveRTSEngine.run_workload()",
      workedExample: "Under the periodic_high_tight benchmark (100 requests arriving every 2.85 ms):\n• Baseline (FIFO + Fixed DeiT-Small): 6.0% on-time completion (collapses due to backlog).\n• Proposed (EDF + AC + Adaptive EViT): 76.0% on-time completion (+70.0% gain).\nProactive admission control rejects 24 infeasible requests, enabling 76 requests to complete on time.",
      nextStepSummary: "Walkthrough complete! Scroll down to inspect the experimental results, comparison tables, and scientific limitations."
    }
  ];

  // =========================================================================
  // DOM REFERENCES
  // =========================================================================
  const el = {
    stepPills: document.getElementById('step-pills'),
    diagramNodes: document.querySelectorAll('.diag-node, .diag-decision-diamond'),
    branchCards: document.querySelectorAll('.branch-card'),
    expStepNum: document.getElementById('exp-step-num'),
    expStepTitle: document.getElementById('exp-step-title'),
    btnPrevStep: document.getElementById('btn-prev-step'),
    btnNextStep: document.getElementById('btn-next-step'),
    btnResetWalkthrough: document.getElementById('btn-reset-walkthrough'),
    explanationBody: document.getElementById('explanation-body'),
    navLinks: document.querySelectorAll('.nav-link')
  };

  let currentStep = 1;
  const totalSteps = STEPS_DATA.length;

  // =========================================================================
  // RENDER STEP FUNCTION
  // =========================================================================
  function renderStep(stepNumber) {
    currentStep = stepNumber;
    const data = STEPS_DATA[stepNumber - 1];

    // 1. Update Step Indicator
    el.expStepNum.textContent = `Step ${data.step} of ${totalSteps}`;
    el.expStepTitle.textContent = data.title.replace(`Step ${data.step}: `, '');

    // 2. Update Nav Buttons State
    el.btnPrevStep.disabled = currentStep === 1;
    el.btnNextStep.textContent = currentStep === totalSteps ? 'Finish Walkthrough' : 'Next Step →';

    // 3. Update Step Selector Pills
    document.querySelectorAll('.step-pill').forEach((pill) => {
      const pStep = parseInt(pill.dataset.step, 10);
      pill.classList.toggle('active', pStep === currentStep);
    });

    // 4. Update Architecture Diagram Node Highlighting
    el.diagramNodes.forEach((node) => {
      const nodeStep = parseInt(node.dataset.step, 10);
      if (data.activeNodeId === "ALL") {
        node.classList.add('active-step');
        node.classList.remove('subdued');
      } else if (nodeStep === currentStep) {
        node.classList.add('active-step');
        node.classList.remove('subdued');
      } else {
        node.classList.remove('active-step');
        node.classList.add('subdued');
      }
    });

    // 5. Update Terminal Branch Cards Highlighting
    el.branchCards.forEach((card) => {
      card.classList.remove('branch-active');
    });
    if (data.activeBranchId) {
      const branchCard = document.getElementById(data.activeBranchId);
      if (branchCard) {
        branchCard.classList.add('branch-active');
      }
    } else if (data.activeNodeId === "ALL") {
      const ontimeCard = document.getElementById('branch-node-ontime');
      if (ontimeCard) ontimeCard.classList.add('branch-active');
    }

    // 6. Build Explanation Body HTML
    let inputsRows = data.inputs.map(i => `
      <tr>
        <td class="io-label">${escapeHtml(i.label)}</td>
        <td class="io-val">${escapeHtml(i.val)}</td>
      </tr>
    `).join('');

    let outputsRows = data.outputs.map(o => `
      <tr>
        <td class="io-label">${escapeHtml(o.label)}</td>
        <td class="io-val">${escapeHtml(o.val)}</td>
      </tr>
    `).join('');

    el.explanationBody.innerHTML = `
      <!-- Left Column: Conceptual & Worked Example -->
      <div class="exp-col-left">
        <div class="exp-block">
          <div class="exp-block-title">
            <span>🔍</span> What Happens at This Stage
          </div>
          <div class="exp-block-content">
            <p>${data.whatHappens.replace(/\n/g, '<br>')}</p>
          </div>
        </div>

        <div class="exp-block">
          <div class="exp-block-title">
            <span>⚙️</span> Why This Step Is Necessary
          </div>
          <div class="exp-block-content">
            <p>${data.whyNecessary}</p>
          </div>
        </div>

        <div class="exp-block">
          <div class="exp-block-title">
            <span>📐</span> Concrete Worked Example
          </div>
          <div class="worked-example-box">
            ${escapeHtml(data.workedExample).replace(/\n/g, '<br>')}
          </div>
        </div>
      </div>

      <!-- Right Column: Formal IO & Source Code Provenance -->
      <div class="exp-col-right">
        <div class="exp-block">
          <div class="exp-block-title">
            <span>📥 / 📤</span> Stage Inputs &amp; Outputs
          </div>
          <table class="io-table">
            <tbody>
              ${inputsRows}
              ${outputsRows}
            </tbody>
          </table>
        </div>

        <div class="source-mapping-card">
          <span class="sm-label">Implementation Source File</span>
          <span class="sm-file">${data.sourceFile}</span>
          <span class="sm-func">${data.sourceMethod}</span>
        </div>

        <div class="exp-block">
          <div class="exp-block-title">
            <span>⏭️</span> Next Stage Preview
          </div>
          <div class="exp-block-content" style="font-size: 0.85rem; color: var(--text-muted);">
            <p>${data.nextStepSummary}</p>
          </div>
        </div>
      </div>
    `;
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // =========================================================================
  // EVENT LISTENERS
  // =========================================================================
  function setupEventListeners() {
    // Step pills click
    el.stepPills.addEventListener('click', (e) => {
      const btn = e.target.closest('.step-pill');
      if (btn) {
        const step = parseInt(btn.dataset.step, 10);
        renderStep(step);
      }
    });

    // Diagram node click
    el.diagramNodes.forEach((node) => {
      node.addEventListener('click', () => {
        const step = parseInt(node.dataset.step, 10);
        if (step && step >= 1 && step <= totalSteps) {
          renderStep(step);
        }
      });
    });

    // Next / Prev / Reset Buttons
    el.btnNextStep.addEventListener('click', () => {
      if (currentStep < totalSteps) {
        renderStep(currentStep + 1);
      } else {
        // Jump to results section
        const resultsSec = document.getElementById('results-limitations');
        if (resultsSec) {
          resultsSec.scrollIntoView({ behavior: 'smooth' });
        }
      }
    });

    el.btnPrevStep.addEventListener('click', () => {
      if (currentStep > 1) {
        renderStep(currentStep - 1);
      }
    });

    el.btnResetWalkthrough.addEventListener('click', () => {
      renderStep(1);
    });

    // Keyboard Arrow Navigation
    window.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' && currentStep < totalSteps) {
        renderStep(currentStep + 1);
      } else if (e.key === 'ArrowLeft' && currentStep > 1) {
        renderStep(currentStep - 1);
      }
    });

    // Navigation Links active state on scroll
    window.addEventListener('scroll', () => {
      const sections = ['overview', 'walkthrough', 'results-limitations'];
      const scrollPos = window.scrollY + 120;

      sections.forEach((id) => {
        const sec = document.getElementById(id);
        const link = document.querySelector(`.nav-link[href="#${id}"]`);
        if (sec && link) {
          const top = sec.offsetTop;
          const height = sec.offsetHeight;
          if (scrollPos >= top && scrollPos < top + height) {
            el.navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
          }
        }
      });
    });
  }

  // Initial render
  setupEventListeners();
  renderStep(1);
});
