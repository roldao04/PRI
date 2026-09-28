# Assignment 2: AI Enhancements Planning Guide

## Understanding the Transition from Lexical to Semantic Search

Your Assignment 1 system represents classical information retrieval, where documents are ranked based on term overlap and statistical properties like term frequency and inverse document frequency. This approach works remarkably well for many queries, but it has fundamental limitations. When a user searches for "inteligência artificial," your BM25 system won't recognize that a document about "aprendizado de máquina" or "redes neurais" might be highly relevant, because there's no lexical overlap. Similarly, when users want direct answers rather than links to read, the traditional IR system provides no mechanism for synthesis or summarization.

Assignment 2 addresses these limitations by introducing neural models that understand meaning rather than just matching words, and language models that can generate natural responses conditioned on retrieved information. The architecture builds on your existing system rather than replacing it, creating a hybrid approach that combines the efficiency of lexical retrieval with the semantic understanding of neural methods.

## Answer Generation: From Retrieval to Response

### The Fundamental Challenge

Answer generation sits at the intersection of information retrieval and natural language generation. Your search engine excels at finding relevant documents, but users increasingly expect direct answers to their questions. The challenge is transforming ranked document lists into coherent, accurate natural language responses without hallucinating information or losing the grounding that makes search engines trustworthy.

The approach you'll implement is called Retrieval-Augmented Generation, which elegantly solves this problem by treating retrieved documents as mandatory context for the language model. Unlike a pure generative approach where the model might fabricate plausible-sounding but incorrect information, RAG constrains the model to synthesize answers from specific source documents. This preserves the reliability of search while adding the natural language interface users want.

### Single-Document RAG: The Foundation Strategy

The most straightforward implementation of RAG uses your top-ranked document as the sole context for answer generation. After your neural reranker identifies the most relevant document, you extract a portion of its content and present it to a large language model alongside the user's query. The LLM then synthesizes an answer based on that specific document.

This approach succeeds or fails based on the quality of your reranking. If the neural reranker reliably places the best document first, single-document RAG produces accurate, focused answers. The implementation requires careful attention to context window management. You need to balance providing enough context for the LLM to find the answer while staying within token limits that affect both cost and latency. A practical strategy extracts the first two to three thousand characters of the document, which captures the introductory material that Wikipedia structures to be definitionally rich.

The prompt design determines answer quality as much as the model choice. Your prompt needs to establish clear boundaries for the LLM. You want to communicate that this is a fact-based task grounded in the provided document, not a creative writing exercise. The prompt should explicitly instruct the model to acknowledge when the document lacks sufficient information rather than speculating or drawing on its training data. This maintains the trustworthiness that users expect from search engines.

### Multi-Document RAG: Enhanced Coverage

Once single-document RAG works reliably, you might consider extending it to synthesize information from multiple documents. This approach addresses a limitation of the single-document strategy where the answer might be distributed across several articles. For instance, a query about "causas da Segunda Guerra Mundial" might benefit from combining information from documents about economic conditions, political movements, and specific triggering events.

The implementation challenge is managing the expanded context window. When you include three to five documents instead of one, you need to truncate each document more aggressively to stay within token limits. This creates a trade-off between breadth and depth. You gain coverage across multiple sources but risk missing details that were deeper in any individual document.

The prompt structure becomes more complex with multiple documents. You need to clearly delineate where each document begins and ends, perhaps using markdown headers or numbered sections. The LLM requires explicit instruction to synthesize across sources rather than simply answering from whichever document it encounters first. You might instruct it to compare perspectives, identify complementary information, or reconcile contradictions between sources.

The primary downside is increased latency and cost. Three documents means roughly three times the tokens in your prompt, which translates directly to slower response times and higher API costs. For your assignment timeline, I recommend implementing this only after single-document RAG works well, and potentially making it an optional feature that users can enable when they want more comprehensive answers.

### Hybrid Extractive-Generative Approach

A sophisticated variation combines extractive question answering with generation. Instead of sending entire document excerpts to the LLM, you first use your neural reranker at a finer granularity to identify the most relevant paragraph or passage within the top document. This passage becomes your focused context for generation.

The advantage is precision. You're feeding the LLM exactly the content that appears most relevant, reducing the chance of distraction from tangential information in the document. This can significantly improve answer quality for longer documents where the answer is embedded deep within technical details or narrative structure. The approach also reduces token consumption because you're sending smaller, more focused context.

The implementation requires extending your reranking pipeline. After document-level reranking, you split the top document into passages and score each passage against the query using your cross-encoder model. The top-scoring passage provides the context for answer generation. This adds another inference step to your pipeline, which increases latency, but the improvement in answer quality often justifies the cost.

### Language Model Selection

Your choice of LLM provider involves balancing several factors including cost, latency, quality, multilingual support, and terms of service. Let me walk through the key considerations for each major option.

Groq offers the fastest inference times in the industry, typically responding in under one second. This creates a dramatically better user experience compared to three to five second response times from other providers. They offer Llama models through their service, which have reasonable Portuguese language support having been trained on substantial multilingual data. The free tier provides thirty requests per minute, which suffices for development and light production use. The limitation is that you're constrained to their model selection, though Llama 3.1 70B is quite capable for factual question answering.

Google Gemini provides excellent multilingual support because Google specifically optimized these models for languages beyond English. Gemini 1.5 Flash offers good performance with a generous context window, though response times are slightly slower than Groq. The free tier is more restrictive at fifteen requests per minute. The instruction-following quality is high, meaning the model generally adheres well to your prompt constraints about staying grounded in the provided document.

OpenAI's GPT-4 Turbo represents the premium option. It delivers the highest quality responses with superior reasoning capabilities and excellent Portuguese support. However, it's significantly more expensive with no free tier, costing roughly one cent per request at your expected prompt size. Response times are also slower. For an academic assignment, this cost structure is difficult to justify unless you need the absolute best answer quality for evaluation purposes.

For your assignment, I strongly recommend starting with Groq. The speed advantage creates a better demonstration, the free tier removes financial barriers, and the quality is sufficient to show that your RAG implementation works correctly. You can always swap providers later because you should abstract the LLM interaction behind a clean interface.

### Information Feeding Strategy

The question of what information to feed the LLM and how much requires thoughtful consideration. Your README should document these decisions because they represent important design choices that affect system behavior.

For single-document RAG, feeding the first two to three thousand characters balances several competing concerns. This length typically captures the lead section and first few substantive sections of a Wikipedia article, where the most important definitional and contextual information resides. It stays well within the context limits of modern LLMs while providing enough information for the model to construct meaningful answers. Longer excerpts risk including tangential information that distracts from the core answer, while shorter excerpts might truncate before reaching the relevant content.

The truncation strategy matters more than you might initially think. Simply cutting the content at a fixed character count might split sentences awkwardly, creating confusion for the LLM. A better approach finds natural boundaries like paragraph breaks or sentence endings near your target length. This ensures the LLM receives coherent, complete thoughts rather than fragments.

Document title inclusion provides crucial context. The title primes the LLM with topical information and helps it understand the domain of the content it's about to process. Including it in your prompt before the document content helps the model better interpret the subsequent text.

### Prompt Engineering Principles

The structure and wording of your prompts dramatically affect answer quality. Let me explain the principles that should guide your prompt design, because this is where much of your system's intelligence actually resides.

The system message establishes the LLM's role and behavioral constraints. This is your opportunity to set expectations about factuality, conciseness, and groundedness in source documents. A well-crafted system message emphasizes that the model is a factual assistant working from Wikipedia articles, not a creative writer or conversationalist. It should explicitly state that answers must derive from the provided document and that admitting insufficient information is preferable to speculation.

The user prompt should follow a consistent structure that helps the LLM understand what you're asking. Present the context first with clear labeling, showing the document title and content. Then state the user's question explicitly. Finally, provide specific instructions about the desired response format. You might specify length expectations, tone preferences, or structural requirements like starting definitions with "X é..." in Portuguese.

Temperature and other sampling parameters control the randomness of the model's output. For factual question answering, you want lower temperatures around 0.3 to 0.4, which bias the model toward high-probability, factual responses. Higher temperatures increase creativity but also increase the risk of hallucination or departing from the source document.

Token limits for the response should be generous enough to allow complete answers but tight enough to prevent rambling. A limit of four to six hundred tokens typically allows two to four paragraph responses, which suits most informational queries. Shorter limits risk truncating explanations, while longer limits can lead to the model padding responses with unnecessary elaboration.

## Further AI Enhancements: Exploring the Options

### Semantic Snippet Extraction: Highlighting Relevance

Semantic snippet extraction represents one of the most immediately valuable enhancements you can implement because it directly improves the search results interface in a way that users immediately recognize and appreciate. Traditional search engines show the first few sentences of each document as a preview, which often provides poor insight into why the document was retrieved or where the relevant information resides. Semantic snippets solve this by showing users the most relevant passages within each document.

The core insight is that your neural reranker can work at multiple levels of granularity. You've already used it to rank entire documents, but the same model can score individual passages within documents. The implementation splits each top-ranked document into coherent passages, scores each passage against the user's query, and extracts the highest-scoring passages as snippets. This shows users exactly where in each document their answer likely resides.

The passage splitting strategy critically affects both the quality of your snippets and the computational cost of scoring. Splitting on paragraph boundaries preserves topical coherence because Wikipedia authors structure content into paragraphs that focus on single ideas. Each paragraph represents a semantically complete unit that can be scored independently. You want passages long enough to be meaningful, typically one hundred to four hundred characters, but short enough that your cross-encoder can process them efficiently within its maximum sequence length.

The scoring process uses the exact same cross-encoder model you use for document reranking, which is why this enhancement is particularly attractive. You're not introducing new dependencies or models to manage. For each document, you create query-passage pairs and score them in a batch, then select the top two or three passages per document. This provides users with multiple entry points into each document's content.

The presentation layer requires thoughtful design to maximize user value. You want to show these snippets prominently in your search results, perhaps in a dedicated section or as highlighted excerpts within the document preview. Adding ellipses before and after snippets signals truncation and helps users understand that they're seeing excerpts rather than complete content. Bolding query terms within snippets helps users quickly verify relevance and understand why this passage scored highly.

The user experience improvement is substantial. Instead of reading generic document previews and clicking through to find relevant sections, users immediately see the most pertinent information. This reduces time to answer significantly, especially for longer Wikipedia articles where relevant content might be buried deep within the document structure. Your README should emphasize this concrete usability benefit.

### Embedding-Based Relevance Feedback: Semantic Document Similarity

Your Assignment 1 system includes a "Find Similar Documents" feature that uses TF-IDF vectors and term overlap to identify related documents. This works reasonably well but has inherent limitations. Documents about conceptually related topics that use different vocabulary appear dissimilar, while documents that happen to share common words might appear similar despite being topically unrelated. Embeddings fundamentally improve this by representing documents in a semantic space where meaning rather than lexical overlap determines similarity.

The key insight behind embeddings is that they compress documents into dense vector representations where semantically similar documents cluster together regardless of their specific word choices. A document about neural networks and a document about deep learning will have similar embeddings even if they share few exact terms, because the embedding model learned during training that these concepts are semantically related. Conversely, documents about Apple Inc. and apple fruit will have distant embeddings despite sharing a word, because the surrounding context signals different meanings.

The implementation challenge that the assignment correctly identifies is scale. For your 1.15 million document collection, you need to compute 1.15 million embeddings. Each embedding is a vector of 384 or 768 floating-point numbers, creating substantial storage requirements of roughly 1.7 gigabytes for 384-dimensional embeddings. More critically, the computation itself is GPU-intensive. CPU-based embedding generation might take fifteen to twenty hours for your full collection, while GPU acceleration reduces this to one to two hours.

The practical approach is to treat embedding generation as a one-time preprocessing step that you perform on GPU infrastructure like Google Colab or Kaggle. You upload your forward index database, run a batch embedding script, and download the resulting embeddings file. This file becomes part of your search engine's data, distributed alongside your inverted index. Users never need to regenerate embeddings unless the document collection changes significantly.

The embedding model selection involves trade-offs between multilingual quality, embedding dimensionality, and encoding speed. Models specifically fine-tuned for multilingual tasks generally perform better for Portuguese content than English-only models. The dimensionality affects both storage requirements and similarity computation speed, with 384 dimensions offering a good balance. Sentence-BERT family models like paraphrase-multilingual-mpnet-base-v2 provide strong multilingual support and have been validated on semantic similarity tasks.

Once you have precomputed embeddings, the runtime similarity search becomes remarkably fast. Finding similar documents is just a vector similarity computation, essentially a dot product for normalized embeddings. This takes milliseconds even for your large collection, which is one to two orders of magnitude faster than your TF-IDF approach that requires loading and comparing term frequency vectors. The speed advantage means you can use semantic similarity as a real-time feature without performance concerns.

The user-facing improvement is that "More Like This" recommendations become genuinely semantic. When users click on a document about machine learning, they get recommendations for documents about neural networks, artificial intelligence, and deep learning, even if these documents use entirely different terminology. The recommendations feel more intelligent because they're based on meaning rather than word overlap.

For your README, you should acknowledge the preprocessing requirement honestly. Explain that initial embedding generation requires GPU access and takes one to two hours, but that this is a one-time cost. The resulting embeddings file can be distributed so that users of your search engine don't need to regenerate embeddings themselves. Document your model choice and explain why semantic similarity provides better recommendations than lexical similarity for exploratory search use cases.

### Query Expansion: Bridging the Vocabulary Gap

Query expansion addresses a fundamental challenge in information retrieval called the vocabulary mismatch problem. Users express information needs using vocabulary that may differ from the terminology in relevant documents. A user searching for "computador" might miss highly relevant documents that use "máquina de computação" or "sistema informático." Query expansion attempts to bridge this gap by automatically adding related terms, synonyms, and alternate phrasings to the original query.

The LLM-based approach to query expansion is elegant and flexible. Instead of relying on hand-crafted synonym dictionaries or statistical co-occurrence analysis, you leverage the language model's understanding of semantic relationships. You send the user's query to your LLM with specific instructions to expand it with relevant terms, and the model generates an enriched query that has better coverage of the relevant vocabulary space.

The implementation requires careful prompt engineering because the expansion strategy significantly affects results. There are several distinct approaches you might take, each with different characteristics. Synonym expansion adds alternate terms with similar meanings, broadening the query's vocabulary coverage without changing its semantic intent. Related concept expansion adds terms from adjacent topics or broader categories, casting a wider net but potentially reducing precision. Query reformulation rewrites the query using more formal or technical vocabulary, which can help when users express needs in colloquial language but documents use technical terminology.

The challenge with query expansion is that it can hurt precision while improving recall. Adding terms increases the chances of matching irrelevant documents that happen to contain those terms in different contexts. This precision-recall trade-off is inherent to expansion strategies and requires careful balancing. Conservative expansion with closely related terms minimizes precision loss, while aggressive expansion maximizes recall at greater precision cost.

A practical implementation strategy makes query expansion optional and transparent. Provide users with a toggle to enable or disable expansion, and show them which terms were added so they understand how their query was modified. This transparency builds trust and helps users understand when expansion helps versus when it introduces noise. Some queries benefit tremendously from expansion, particularly short, underspecified queries like "animais marinhos" that might miss documents using specific taxonomic terms. Other queries, especially specific proper nouns or technical phrases, don't need expansion and might be harmed by it.

The evaluation strategy should demonstrate when expansion helps. Create test cases that show precision improvement for ambiguous queries and recall improvement for underspecified queries. Document cases where expansion doesn't help or hurts results, showing that you understand the trade-offs involved. Your README should explain the precision-recall balance and discuss when expansion is most beneficial.

### Additional Creative Enhancement: Search Quality Signals

Beyond the three suggested enhancements, you might consider implementing search quality signals that provide transparency about result confidence. This involves augmenting search results with metadata that helps users assess reliability and relevance. You could show the neural reranker's confidence score alongside each result, giving users insight into how certain the system is about each ranking. You could display the number of query terms matched per document, helping users understand whether results address their entire query or just fragments of it.

Another creative direction is implementing simple personalization based on search session history. Track which documents users click on during a session, identify common topics or categories among those documents, and slightly boost similar documents in subsequent searches. This doesn't require complex user profiles or privacy-invasive tracking, just session-based preference learning that expires when the user closes their browser.

## Implementation Timeline and Priorities

Given your deadline pressure, strategic prioritization becomes critical. The neural reranker represents your highest-priority task because it's worth the most points and serves as the foundation for other enhancements. Get this working first and working well. Single-document RAG comes next because it's worth substantial points and provides clear user value. Semantic snippet extraction makes an excellent third priority because it reuses your reranker model and creates obvious UX improvements. Query expansion can serve as a fourth enhancement if time permits, as it's relatively quick to implement once you have your RAG system working.

Embedding-based similarity, despite its technical elegance, presents timeline risk due to the GPU preprocessing requirement. The one to two hour computation time isn't the issue, but coordinating GPU access, debugging the embedding pipeline, and validating results adds complexity when you're under deadline pressure. Your existing TF-IDF similarity from Assignment 1 already works reasonably well. Consider implementing semantic similarity only if you have comfortable time margins after completing the higher-priority enhancements.

Each enhancement should be implemented with proper abstraction so you can test it independently and toggle it on or off in your API. This modularity helps with debugging and allows you to demonstrate progressive enhancement during evaluation. Your README should document what you implemented, what you chose not to implement, and the reasoning behind those decisions. Showing that you understand trade-offs and made strategic choices demonstrates maturity even if you didn't implement every possible feature.
