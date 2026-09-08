"""Builds the Assignment 2 Word report: bare minimum write up, real result tables,
screenshot placeholders, no stylistic hyphens, no colored headings/tables."""
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = "/Users/anees/Documents/Coursework/data266-5330/assignments/assignment-2/HW2_Document.docx"

doc = docx.Document()

# remove theme blue from Title / Heading 1 / Heading 2
for style_name in ("Title", "Heading 1", "Heading 2"):
    style = doc.styles[style_name]
    style.font.color.rgb = RGBColor(0, 0, 0)


def set_cell_shading(cell, color_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color_hex)
    tcPr.append(shd)


def add_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "999999")
        borders.append(el)
    tblPr.append(borders)


def add_table(rows, header=True):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    add_table_borders(table)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = str(val)
            for para in cell.paragraphs:
                for r in para.runs:
                    r.font.size = Pt(9.5)
                    r.font.color.rgb = RGBColor(0, 0, 0)
                    if header and i == 0:
                        r.font.bold = True
            if header and i == 0:
                set_cell_shading(cell, "D9D9D9")
    doc.add_paragraph()
    return table


def add_screenshot_placeholder(description):
    table = doc.add_table(rows=1, cols=1)
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "dashed")
        el.set(qn("w:sz"), "8")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "888888")
        borders.append(el)
    tblPr.append(borders)
    cell = table.rows[0].cells[0]
    set_cell_shading(cell, "F2F2F2")
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(14)
    run = p.add_run(f"[SCREENSHOT PLACEHOLDER]\n{description}")
    run.italic = True
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    run.font.size = Pt(10)
    doc.add_paragraph()


def h1(text):
    doc.add_heading(text, level=1)


def h2(text):
    doc.add_heading(text, level=2)


def p(text):
    doc.add_paragraph(text)


def code(text):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    doc.add_paragraph()


A = "→"  # plain arrow, not a hyphen, used for "before to after" values

# ============================================================ Title
doc.add_heading("DATA 266 HW2", level=0)
doc.styles["Title"].font.color.rgb = RGBColor(0, 0, 0)
p("Embedding transfer learning on IMDB reviews, a Retrieval Augmented Generation pipeline over "
  "10 movie Wikipedia pages, and five training time optimization techniques.")

# ============================================================ Part 1
h1("1. Embedding Transfer Learning (IMDB Reviews)")

h2("1.1 Setup")
p("Pretrained model: word2vec-google-news-300 (3,000,000 words, 300 dimensions), loaded with "
  "gensim. Finetuning data: 15,000 reviews sampled from the IMDB Movie Reviews dataset "
  "(stanfordnlp/imdb), cleaned of HTML tags and tokenized with gensim simple_preprocess. A new "
  "gensim Word2Vec model (skip gram, 300 dimensions) was built on the IMDB vocabulary. Words "
  "shared with the pretrained vocabulary were initialized with their pretrained vectors "
  "(20,040 of 22,723 words matched), then trained for 5 epochs on the IMDB corpus.")

h2("1.2 Top 3 nearest neighbors before finetuning")
add_table([
    ["Word", "Rank", "Neighbor", "Cosine similarity"],
    ["cast", 1, "casts", 0.7219],
    ["cast", 2, "casting", 0.7188],
    ["cast", 3, "Cast", 0.6638],
    ["score", 1, "scoring", 0.7197],
    ["score", 2, "scores", 0.6596],
    ["score", 3, "scored", 0.6384],
    ["plot", 1, "plots", 0.7625],
    ["plot", 2, "Plot", 0.6524],
    ["plot", 3, "plotting", 0.6328],
    ["screen", 1, "screens", 0.7729],
    ["screen", 2, "onscreen", 0.6115],
    ["screen", 3, "LCD_screen", 0.5599],
    ["review", 1, "reviewed", 0.6630],
    ["review", 2, "reviewing", 0.6610],
    ["review", 3, "reviews", 0.6380],
])
add_screenshot_placeholder("Notebook cell showing the pretrained model neighbor table "
                            "(Part 1, Section 1.2).")

h2("1.3 Top 3 nearest neighbors after finetuning")
add_table([
    ["Word", "Rank", "Neighbor", "Cosine similarity"],
    ["cast", 1, "supporting", 0.5975],
    ["cast", 2, "ensemble", 0.5629],
    ["cast", 3, "casts", 0.5266],
    ["score", 1, "ennio", 0.5714],
    ["score", 2, "morricone", 0.5708],
    ["score", 3, "scoring", 0.5438],
    ["plot", 1, "story", 0.6103],
    ["plot", 2, "storyline", 0.5662],
    ["plot", 3, "plotline", 0.5534],
    ["screen", 1, "screens", 0.5327],
    ["screen", 2, "onscreen", 0.5216],
    ["screen", 3, "kapadia", 0.3945],
    ["review", 1, "reviews", 0.6052],
    ["review", 2, "comment", 0.5809],
    ["review", 3, "comments", 0.5578],
])
p("Neighbors shift toward the film sense of each word after finetuning: cast toward supporting "
  "and ensemble, score toward ennio and morricone (the composer Ennio Morricone), plot toward "
  "story and storyline, and review toward reviews and comment. screen changes the least since "
  "it was already film related before finetuning.")
add_screenshot_placeholder("Notebook cell showing the finetuned model neighbor table and the "
                            "finetuning training log (Part 1, Sections 1.3 and 1.4).")

h2("1.4 2D visualization of the embedding shift")
p("Figure 1 shows all 5 words before and after finetuning in 2D using t SNE. Figure 2 zooms in "
  "on plot and its top 3 neighbors in both spaces. plot moves away from plots and plotting and "
  "toward story and storyline.")
add_screenshot_placeholder("Figure 1, t SNE plot of all 5 words before and after finetuning "
                            "(Part 1, Section 1.7).")
add_screenshot_placeholder("Figure 2, t SNE detail plot of plot and its neighbors, before and "
                            "after finetuning (Part 1, Section 1.7).")

h2("1.5 Cosine similarity of each word, original vs finetuned")
add_table([
    ["Word", "Cosine similarity (original vs finetuned)"],
    ["review", 0.5757],
    ["score", 0.6441],
    ["screen", 0.6652],
    ["plot", 0.6738],
    ["cast", 0.6969],
])
p("Most shifted word: review (cosine similarity 0.5757). review has many general senses "
  "(performance review, code review) so the narrow IMDB usage pulls it furthest from its "
  "pretrained position.")
p("Least shifted word: cast (cosine similarity 0.6969). cast already carries a strong film "
  "sense in general English, so IMDB usage reinforces rather than changes its meaning.")
add_screenshot_placeholder("Notebook cell showing the cosine similarity table and the most "
                            "shifted / least shifted print statement (Part 1, Section 1.8).")

# ============================================================ Part 2
h1("2. Retrieval Augmented Generation (RAG) Pipeline")

h2("2.1 Setup")
p("Corpus: Wikipedia pages for 10 movies, loaded with LangChain DirectoryLoader and "
  "TextLoader. Splitter: RecursiveCharacterTextSplitter, chunk size 500, chunk overlap 50, "
  "giving 1,970 chunks. Embeddings: sentence transformers/all MiniLM L6 v2. Vector store: "
  "FAISS, retriever returns the top 3 chunks. LLM: Qwen2.5 0.5B Instruct, run locally with "
  "greedy decoding for reproducible answers. The pipeline uses explicit steps, a prompt "
  "template, a retriever, and a direct LLM call, instead of one prebuilt chain.")
add_screenshot_placeholder("Notebook cell showing the 10 loaded documents and the chunk count "
                            "after splitting (Part 2, Sections 2.1 to 2.3).")

h2("2.2 Five questions, retrieved chunks, and generated answers (chunk size 500, overlap 50)")
add_table([
    ["#", "Question", "Top retrieved chunk source (of 3)", "Generated answer", "Correct"],
    [1, "Who directed Inception and who composed its music?", "Inception.txt",
     "Christopher Nolan directed Inception, while Hans Zimmer composed its music.", "Yes"],
    [2, "What is the profession of Andy Dufresne before he was imprisoned in The Shawshank "
        "Redemption?", "The_Shawshank_Redemption.txt",
     "Before being imprisoned in The Shawshank Redemption, Andy Dufresne worked as a banker.",
     "Yes"],
    [3, "What ship does the film Titanic depict sinking, and in what year did it sink?",
     "Titanic_1997.txt", "The film Titanic depicts the sinking of the RMS Titanic in 1912.",
     "Yes"],
    [4, "In The Matrix, what is the name of the character played by Keanu Reeves?",
     "The Matrix.txt", "Keanu Reeves as Neo", "Yes"],
    [5, "What dinosaur species famously breaks out of its paddock in Jurassic Park?",
     "Jurassic_Park.txt", "The dinosaur species that breaks out of its paddock in Jurassic "
     "Park is the Velociraptor.", "No, correct answer is Tyrannosaurus rex"],
])
add_screenshot_placeholder("Notebook cell showing all 5 questions with their top 3 retrieved "
                            "chunks and generated answers, chunk size 500 overlap 50 "
                            "(Part 2, Section 2.5).")

h2("2.3 Changed chunk size and overlap for 2 of the 5 questions")
p("The vector store was rebuilt with chunk size 1000, chunk overlap 100 (1,112 chunks instead "
  "of 1,970), and the Jurassic Park and Titanic questions were run again.")
add_table([
    ["Question", "Original, size 500 overlap 50", "Changed, size 1000 overlap 100"],
    ["Jurassic Park paddock dinosaur",
     "Top 3 chunks miss the Tyrannosaurus rex sentence. An earlier Velociraptor scene is "
     "retrieved instead. Answer: Velociraptor, wrong.",
     "The top chunk now contains the Tyrannosaurus rex sentence. Answer: Dilophosaurus, still "
     "wrong, a different dinosaur named two sentences later in the same chunk."],
    ["Titanic ship and year",
     "Top chunk contains the sinking of RMS Titanic in 1912. Answer correct.",
     "Same fact still in the top chunk. Answer correct, unchanged."],
])
p("Bigger chunks fixed retrieval for the Jurassic Park question, but the final answer was still "
  "wrong. The Titanic question was already correct at 500/50 and stayed correct at 1000/100.")
add_screenshot_placeholder("Notebook cell showing the rebuilt chunk count and the new retrieved "
                            "chunks and answers for the Jurassic Park and Titanic questions "
                            "(Part 2, Section 2.6).")

h2("2.4 Retrieval success rate (chunk size 500, overlap 50)")
add_table([
    ["Question", "Ground truth passage", "Found in top 3", "Rank of first relevant chunk"],
    ["Inception director and composer",
     "directed by Christopher Nolan / score composed by Hans Zimmer", "Yes", 1],
    ["Andy Dufresne profession", "banker Andy Dufresne arrives at Shawshank State Prison",
     "Yes", 1],
    ["Titanic ship and year", "sinking of RMS Titanic in 1912", "Yes", 1],
    ["Matrix character played by Keanu Reeves", "Keanu Reeves as Neo", "Yes", 1],
    ["Jurassic Park paddock dinosaur",
     "allows a Tyrannosaurus rex to escape and attack the touring group", "No", "N/A"],
])
p("Retrieval success rate: 4 of 5, 80 percent.")
add_screenshot_placeholder("Notebook cell showing the retrieval success table and the "
                            "computed rate (Part 2, Section 2.7).")

h2("2.5 RAG failure analysis")
p("Failure 1: retrieval miss causes a wrong answer. At chunk size 500, the sentence about the "
  "Tyrannosaurus rex escaping its paddock is not in the top 3 chunks. The retriever returns the "
  "page title, a list header, and an earlier scene mentioning a Velociraptor. The model answers "
  "Velociraptor, which is wrong.")
p("Failure 2: correct context retrieved, wrong answer generated. At chunk size 1000, the "
  "correct sentence about the Tyrannosaurus rex is in the top chunk. The model still answers "
  "Dilophosaurus, a different dinosaur named two sentences later in the same chunk. This shows "
  "that fixing retrieval does not guarantee a correct answer.")
p("Both failures come from several dinosaur names appearing close together in one passage, "
  "which confuses both retrieval and generation.")

# ============================================================ Part 3
h1("3. Training Time Optimization Techniques")

h2("3.1 Shared setup")
p("Model: an 8 layer Transformer encoder classifier, 384 dimensions, 6 heads. Data: one fixed "
  "random batch, batch size 64, sequence length 256, reused for every experiment. Training: 20 "
  "steps per run on the same batch. Hardware: Apple Silicon GPU, MPS backend, no CUDA "
  "available. Memory was read with torch.mps.current_allocated_memory(), sampled right after "
  "the forward pass, since CUDA memory functions are not available on this machine.")

h2("3.2 Tensor creation, CPU vs GPU")
code('x_cpu = torch.randn(64, 256, 384)                 # CPU tensor\n'
     'x_gpu = torch.randn(64, 256, 384, device="mps")    # GPU tensor')
add_table([
    ["Device", "Time, 20 steps", "Peak memory", f"Loss, start {A} end"],
    ["CPU", "67.63 s", "7783.9 MB", f"0.7021 {A} 0.6972"],
    ["MPS (GPU)", "32.82 s", "7078.0 MB", f"0.7004 {A} 0.6858"],
])
p("MPS was 2.06 times faster than CPU for the same model, data, and steps. Final loss values "
  "are close, so both runs did the same computation on different hardware. The two memory "
  "numbers are not directly comparable: CPU memory is whole process memory, MPS memory is the "
  "allocator's live tensor count.")
add_screenshot_placeholder("Notebook cell showing the CPU and MPS time, memory, and loss "
                            "output, and the computed speedup (Part 3, Section 1).")

h2("3.3 Weight initialization")
code('nn.init.xavier_uniform_(layer.weight)   # Xavier\n'
     'nn.init.zeros_(layer.weight)             # all zeros baseline')
add_table([
    ["Init scheme", "Time, 20 steps", "Peak memory", f"Loss, start {A} end"],
    ["default (Kaiming)", "42.87 s", "7077.4 MB", f"0.7004 {A} 0.6858"],
    ["Xavier", "43.19 s", "7079.1 MB", f"0.7439 {A} 0.7452"],
    ["zeros", "43.60 s", "7078.1 MB", f"0.6931 {A} 0.3431"],
])
p("Default init decreases smoothly. Xavier starts higher and does not recover within 20 steps. "
  "Zeros still trains because of residual connections, which let gradients skip the zeroed "
  "layers.")
add_screenshot_placeholder("Notebook cell showing the three init schemes' time, memory, loss "
                            "output, and the loss over steps plot (Part 3, Section 2).")

h2("3.4 Activation checkpointing")
code("x = layer(x)                                             # normal\n"
     "x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)  # checkpointed")
add_table([
    ["Configuration", "Time, 20 steps", "Peak activation memory", f"Loss, start {A} end"],
    ["No checkpointing", "47.68 s", "7078.5 MB", f"0.7004 {A} 0.6858"],
    ["With checkpointing", "52.39 s", "357.1 MB", f"0.7004 {A} 0.6886"],
])
p("Checkpointing cuts peak activation memory by about 20 times, at the cost of about 10 "
  "percent more time, since checkpointed layers are recomputed during backward.")
add_screenshot_placeholder("Notebook cell showing the with and without checkpointing time, "
                            "memory, and loss output (Part 3, Section 3).")

h2("3.5 Gradient accumulation")
code("optimizer.zero_grad()\n"
     "for micro_x, micro_y in microbatches:      # 4 microbatches of size 16\n"
     "    out = model(micro_x)\n"
     "    loss = loss_fn(out, micro_y) / 4\n"
     "    loss.backward()\n"
     "optimizer.step()")
add_table([
    ["Configuration", "Time, 20 steps", "Peak memory", f"Loss, start {A} end"],
    ["Baseline, batch 64", "37.29 s", "7078.9 MB", f"0.7004 {A} 0.6858"],
    ["4 microbatches of 16", "37.04 s", "1971.2 MB", f"0.6989 {A} 0.6908"],
])
p("Splitting the batch into 4 microbatches of 16 cuts peak memory by about 3.6 times, with "
  "almost no change in time and a nearly identical loss curve.")
add_screenshot_placeholder("Notebook cell showing the baseline and gradient accumulation time, "
                            "memory, and loss output (Part 3, Section 4).")

h2("3.6 Mixed precision training")
code('with torch.autocast(device_type="mps", dtype=torch.float16):\n'
     "    out = model(x)\n"
     "    loss = loss_fn(out, y)\n"
     "loss.backward()")
add_table([
    ["Precision", "Time, 20 steps", "Peak memory", f"Loss, start {A} end"],
    ["fp32 (baseline)", "37.46 s", "7081.1 MB", f"0.7004 {A} 0.6858"],
    ["fp16 autocast", "35.09 s", "5125.0 MB", f"0.7004 {A} 0.6853"],
])
p("fp16 autocast cuts peak memory by about 28 percent and gives a small speed gain here, with "
  "final loss matching the fp32 baseline. Mixed precision speedups are usually larger on CUDA "
  "GPUs with tensor cores.")
add_screenshot_placeholder("Notebook cell showing the fp32 and mixed precision time, memory, "
                            "and loss output (Part 3, Section 5).")

h2("3.7 Summary")
add_table([
    ["Technique", "Config A", "Time A", "Mem A (MB)", "Loss A",
     "Config B", "Time B", "Mem B (MB)", "Loss B"],
    ["Tensor creation", "CPU", "67.63 s", 7783.9, 0.6972, "MPS", "32.82 s", 7078.0, 0.6858],
    ["Weight init default", "default", "42.87 s", 7077.4, 0.6858, "N/A", "N/A", "N/A", "N/A"],
    ["Weight init xavier", "xavier", "43.19 s", 7079.1, 0.7452, "N/A", "N/A", "N/A", "N/A"],
    ["Weight init zeros", "zeros", "43.60 s", 7078.1, 0.3431, "N/A", "N/A", "N/A", "N/A"],
    ["Checkpointing", "no checkpoint", "47.68 s", 7078.5, 0.6858,
     "checkpoint", "52.39 s", 357.1, 0.6886],
    ["Gradient accumulation", "batch 64", "37.29 s", 7078.9, 0.6858,
     "4x16 microbatch", "37.04 s", 1971.2, 0.6908],
    ["Mixed precision", "fp32", "37.46 s", 7081.1, 0.6858,
     "fp16", "35.09 s", 5125.0, 0.6853],
])
p("Tensor placement, CPU vs GPU, gave the largest speedup. Weight init only changes the loss "
  "curve, not memory or time. Checkpointing and gradient accumulation both trade a small amount "
  "of time for a large memory saving. Mixed precision gives a smaller memory and speed gain on "
  "this hardware than it would on a CUDA GPU.")
add_screenshot_placeholder("Notebook cell showing the final summary table across all 5 "
                            "techniques (Part 3, Summary section).")

# ============================================================ AI Use
h1("4. AI Use")
p("See AI_USE.md.")

doc.save(OUT)
print("wrote", OUT)
