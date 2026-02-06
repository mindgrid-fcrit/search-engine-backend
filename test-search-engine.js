import fs from 'fs';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import 'dotenv/config';

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: "text-embedding-004" });

// --- CONFIGURATION ---
const BATCH_SIZE = 10;
const OUTPUT_FILE = './benchmark_results.json';
const TEST_QUERIES = [
    "How to optimize Python code?", // Technical
    "Help me write a professional email", // Business
    "Creative story about a lost king", // Narrative
    "Excel formulas for complex math", // Specialized
    "Debugging javascript memory leaks", // Debugging
    "Career advice for junior developers", // Roleplay
    "Latex code for tables", // Tools
    "Marketing strategy for software", // Strategy
    "Interpret a dream about flying", // Symbolic
    "Side scrolling game logic" // Logic
];

async function runBenchmark() {
    console.log(`🚀 Starting Benchmarking (Batch size: ${BATCH_SIZE})...`);
    let results = [];

    for (let i = 0; i < BATCH_SIZE; i++) {
        const query = TEST_QUERIES[i];
        console.log(`[${i+1}/${BATCH_SIZE}] Testing: "${query}"`);

        // 1. Generate Embedding (for Semantic Search)
        const embedStart = performance.now();
        const res = await embedModel.embedContent(query);
        const vector = res.embedding.values;
        const embedTime = performance.now() - embedStart;

        // 2. Test Semantic Search (Your Hybrid Model)
        const semStart = performance.now();
        const { data: semData } = await supabase.rpc('hybrid_search_prompts', {
            query_embedding: vector,
            alpha_weight: 0.7,
            match_count: 5
        });
        const semTime = performance.now() - semStart;

        // 3. Test Keyword Search (Baseline)
        const keyStart = performance.now();
        const { data: keyData } = await supabase.rpc('keyword_search_prompts', {
            query_text: query,
            match_count: 5
        });
        const keyTime = performance.now() - keyStart;

        // 4. Structure Data
        results.push({
            query,
            metrics: {
                embedding_latency_ms: embedTime.toFixed(2),
                semantic_db_latency_ms: semTime.toFixed(2),
                keyword_db_latency_ms: keyTime.toFixed(2),
                total_semantic_latency_ms: (embedTime + semTime).toFixed(2)
            },
            results: {
                semantic_top_id: semData?.[0]?.id || "none",
                keyword_top_id: keyData?.[0]?.id || "none"
            }
        });
    }

    // Save to File
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2));
    console.log(`✅ Results saved to ${OUTPUT_FILE}`);
}

runBenchmark().catch(console.error);