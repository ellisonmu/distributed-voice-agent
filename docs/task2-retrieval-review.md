# Task 2: inspection of saved retrieval passages

This is a preliminary assistant inspection of existing Perlmutter output, not a new
experiment, a clinical assessment, a human grade, or an answer-accuracy benchmark.
Thomas's confirmation is pending. Assessments concern whether the retrieved text supports
the question; they do not establish medical correctness or agreement with MedQuAD answers.

## Evidence

- Baseline: 2,047 chunks; candidate: 5,689 chunks; both use BGE-small with dimension 384.
- Ten fixed dev questions, top-3 passages per index; candidate allocation `61464498`.
- Saved report: `manifests/rag-genetics-comparison.json` under the Perlmutter data snapshot.
- Local copy: `logs/rag-genetics-comparison-61464498.json` (ignored by Git).
- The baseline and candidate passages were inspected. No generated answers,
  reference-answer grades or additional retrieval runs were used.

## Passage-support findings

| Question topic | Baseline top-3 | Candidate top-3 | Evidence and limits |
|---|---|---|---|
| Epidermolysis bullosa acquisita symptoms | No support for the named condition | No support for the named condition | Candidate passages discuss simplex, dystrophic and pyloric-atresia forms. Their blistering descriptions do not establish symptoms for the specifically requested acquisita condition. Similar names are insufficient. |
| Volcanoes information | Supports the broad information request | Same support | The same first passage describes eruptions and their effects. Additional health-information evaluation passages are off-topic. |
| Anonychia congenita symptoms | No support for the named condition | Direct support at rank 1 | Candidate text describes missing or partly missing nails, onset at birth and affected fingers/toes. The other two passages concern pachyonychia, not the requested condition. |
| Lynch syndrome treatments | No support | Topic match at rank 2, but no treatment support | The new Lynch passage describes cancer risks and polyps. It does not describe treatment. The other candidate passages concern myasthenia and carpal tunnel treatment. Moving the Lynch passage first would not itself supply treatment evidence. |
| RAPADILINO syndrome inheritance | No support for the named condition | Topic match at ranks 1–2, but no explicit inheritance support | The RAPADILINO passages describe features and mention gene mutations, without stating whether/how the condition is inherited. The third passage describes another disorder. A mutation statement alone does not establish an inheritance pattern. |
| Ornithine transcarbamylase deficiency genetic changes | No support for the named condition | No support for the named condition | Candidate texts concern carbamoyl phosphate synthetase I deficiency, beta-ureidopropionase deficiency and glutaric acidemia type II. They do not establish genetic changes for the requested condition. |
| Hereditary hemochromatosis inheritance | Supports the question at rank 1 | Supports it at rank 3 | The unchanged Hemochromatosis passage explicitly states that primary hemochromatosis is inherited. |
| Glomerular Diseases definition | Insufficient for the requested definition | Partial background, not a complete definition | Baseline passages concern kidney testing/cysts/disease generally. Candidate rank 2 describes glomeruli and one specific glomerulopathy; the other passages concern different kidney conditions. |
| Cancer Chemotherapy information | Supports the broad question | Same support | The matching passages explain chemotherapy, its purpose, side effects and administration. |
| Myelodysplastic Syndromes definition | Supports the question at rank 1 | Supports it at rank 2 | Candidate rank 1 describes the narrower 5q-minus subtype; the original broad MDS overview is retained at rank 2. Do not generalize subtype details to all MDS. |

## Remaining checks

Ranking changes alone cannot repair missing treatment or inheritance evidence. The sample
is small and was already used to guide corpus development; do not report its assessments
as held-out performance or compute an accuracy claim from this table.

The compared index used `text-list/text/html` and omitted `inheritance-pattern-list` and
`related-gene-list`. Subsequent inspection of the actual RAPADILINO record confirmed
an `Autosomal recessive` inheritance memo and a `RECQL4` related-gene entry. The
hemochromatosis record lists two inheritance memos and multiple genes, without a
per-gene inheritance mapping in those lists.

An optional `--include-genetics-metadata` parser mode has been added locally to render the
publisher's labels and associations as separate source-metadata passages. Code-only labels
are not guessed, multiple patterns are preserved, and gene lists are not turned into
specific-variant claims. Thomas's local corpus/parser/audit fixture suite passed all 54 tests
in 0.14 s (`logs/task2-genetics-metadata.xml`), followed by successful real preparation.
The narrative-only results above remain preserved.

## Metadata-enabled comparison

The third index built in allocation `61466521` with 8,227 vectors of dimension 384 and
passed its integrity checks. `manifests/rag-metadata-comparison.json` stores all three
indexes' passages for the same ten dev questions. The pasted comparison provides these
qualitative findings; the full new report has not been independently human graded.

| Query | Metadata-enabled result | Support assessment |
|---|---|---|
| RAPADILINO inheritance | Inheritance-pattern metadata ranks first; its excerpt contains the publisher's Autosomal recessive label | Supplies the previously omitted inheritance evidence for this probe |
| Hereditary hemochromatosis inheritance | Inheritance-pattern metadata ranks first and lists Autosomal recessive and Autosomal dominant | Supports reporting the source's listed patterns; no per-gene/subtype mapping is established |
| Lynch syndrome treatments | Myasthenia Gravis stays first; Lynch related-gene metadata is second and its description third | Gene associations and the retrieved description do not supply treatment evidence |
| Ornithine transcarbamylase deficiency genetic changes | Other conditions' related-gene metadata now occupies top-3 | The requested condition remains unsupported despite higher similarity scores |
| Epidermolysis bullosa acquisita | The same three differently named epidermolysis conditions remain | The earlier exact-condition support problem remains |
| Other probes | Volcanoes, Anonychia, Glomerular Diseases, Chemotherapy and MDS retain their displayed Genetics-only top-3 titles and scores | Preserve the earlier per-question assessments |

This diagnoses a source-extraction omission and demonstrates its effect on two specific
development probes. The same queries guided corpus changes, so these findings are not
held-out performance or a general accuracy estimate.

Keep both frozen indexes and the original comparison report. Any parser/retriever change
needs its own output directory and comparison. Human review, broader validation and the
course project's real answer-quality evaluation remain separate work.
