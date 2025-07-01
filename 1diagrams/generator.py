import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set context for publication-quality plots
sns.set_context("talk")

# Data Definitions
tasks = ['Conversational Chatbot', 'Long Context Handling', 'Code Completion']
perplexity_baseline = [18.2, 19.0, 17.8]
perplexity_tiered = [17.1, 17.6, 16.9]

rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
rouge_baseline = [0.42, 0.28, 0.38]
rouge_tiered = [0.48, 0.32, 0.44]

depths = ['10%', '50%', '90%']
em_baseline = [0.92, 0.78, 0.55]
em_tiered = [0.92, 0.83, 0.70]

tiers = ['Hot', 'Warm', 'Cold']
compression_ratios = [1.5, 6, 35]

retrieval_systems = ['RAG', 'Tiered']
retrieval_times = [85, 50]

faithfulness = [0.77, 0.88]
bleu = [0.55, 0.63]
models = ['Baseline', 'Tiered']

width = 0.35  # bar width

# === Group 1: Perplexity & ROUGE ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
fig.suptitle('Figure 4.1: Language Modeling & Summarization', fontsize=16, y=1.02)

# (a) Perplexity Comparison
x = np.arange(len(tasks))
axs[0].plot(x, perplexity_baseline, marker='o', label='Baseline')
axs[0].plot(x, perplexity_tiered, marker='o', label='Selective Compression')
axs[0].set_title('(a) Perplexity Comparison')
axs[0].set_xticks(x)
axs[0].set_xticklabels(tasks, rotation=15, ha='right')
axs[0].set_ylabel('Perplexity')
axs[0].legend()

# (b) Summarization ROUGE Scores
x = np.arange(len(rouge_metrics))
axs[1].bar(x - width/2, rouge_baseline, width, label='Baseline')
axs[1].bar(x + width/2, rouge_tiered, width, label='Selective Compression')
axs[1].set_title('(b) Summarization ROUGE Scores')
axs[1].set_xticks(x)
axs[1].set_xticklabels(rouge_metrics)
axs[1].set_ylabel('ROUGE Score')
axs[1].legend()

plt.tight_layout()
plt.savefig("1diagrams/group1_perplexity_rouge_sidebyside.png")
plt.close()


# === Group 2: QA EM & Compression Ratio ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
fig.suptitle('Figure 4.2: Long-Context QA & Memory Compression', fontsize=16, y=1.02)

# (a) Exact-Match QA Accuracy at Context Depth
x = np.arange(len(depths))
axs[0].plot(x, em_baseline, marker='s', label='Baseline')
axs[0].plot(x, em_tiered, marker='s', label='Selective Compression')
axs[0].set_title('(a) QA EM vs. Context Depth')
axs[0].set_xticks(x)
axs[0].set_xticklabels(depths)
axs[0].set_ylabel('Exact-Match Accuracy')
axs[0].legend()

# (b) Compression Ratio by Memory Tier
x = np.arange(len(tiers))
axs[1].bar(x, compression_ratios, color='steelblue')
axs[1].set_title('(b) Compression Ratios by Tier')
axs[1].set_xticks(x)
axs[1].set_xticklabels(tiers)
axs[1].set_ylabel('Compression Ratio (×)')
axs[1].set_ylim(0, max(compression_ratios)*1.1)

plt.tight_layout()
plt.savefig("1diagrams/group2_qa_compression_sidebyside.png")
plt.close()


# === Group 3: Retrieval Speed & Faithfulness ===
fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
fig.suptitle('Figure 4.3: Retrieval Latency & Generative Fidelity', fontsize=16, y=1.02)

# (a) Retrieval Time Comparison
axs[0].bar(retrieval_systems, retrieval_times, color='orchid')
axs[0].set_title('(a) Retrieval Time Comparison')
axs[0].set_ylabel('Time (ms)')
axs[0].set_ylim(0, max(retrieval_times)*1.1)

# (b) Faithfulness & Coherence
x = np.arange(len(models))
axs[1].bar(x - width/2, faithfulness, width, label='Faithfulness')
axs[1].bar(x + width/2, bleu, width, label='BLEU Coherence')
axs[1].set_title('(b) Faithfulness & Coherence')
axs[1].set_xticks(x)
axs[1].set_xticklabels(models)
axs[1].set_ylabel('Score')
axs[1].legend()
axs[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("1diagrams/group3_retrieval_faithfulness_sidebyside.png")
plt.close()
