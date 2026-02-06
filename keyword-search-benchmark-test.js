import fs from 'fs';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import crypto from 'crypto';
import 'dotenv/config';

// 1. SETUP (Using your working syntax)
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

const KEYWORD_TEST_CASES = [
    "Python", "Email", "Dream", "LaTeX", "Fantasy", 
    "Game", "Bugs", "Parallel", "Excel", "Career"
];

// 2. VECTOR & CACHE LOGIC (Exact copy from your working code)
async function getVectorWithCache(text) {
    const hash = crypto.createHash('md5').update(text.trim().toLowerCase()).digest('hex');
    const startCache = performance.now();
    
    const { data: cached } = await supabase.from('search_cache').select('embedding').eq('id', hash).single();
    
    if (cached) {
        const vector = typeof cached.embedding === 'string' ? JSON.parse(cached.embedding) : cached.embedding;
        return { embedding: vector, cacheHit: true, time: performance.now() - startCache };
    }

    const result = await embedModel.embedContent(text);
    const embedding = result.embedding.values;
    
    await supabase.from('search_cache').upsert({ id: hash, query_text: text, embedding });
    return { embedding, cacheHit: false, time: performance.now() - startCache };
}

// 3. MAIN BENCHMARK RUNNER
async function runBenchmark() {
    console.log("🔥 WARMING UP: Priming DB and Server connection...");
    // Warmup using your working strategy
    const warmupVec = await getVectorWithCache("warmup");
    await supabase.rpc('hybrid_search_prompts', { query_embedding: warmupVec.embedding, match_count: 1 });
    console.log("✅ Warm-up complete. Starting 10-case test...\n");

    let finalResults = [];

    for (const word of KEYWORD_TEST_CASES) {
        process.stdout.write(`🔎 Testing: [${word.padEnd(10)}] `);

        try {
            // A. Get Vector (Cached or API)
            const embedData = await getVectorWithCache(word);

            // B. Semantic/Hybrid Search (Targeting your RPC)
            const startSem = performance.now();
            const { data: semData } = await supabase.rpc('hybrid_search_prompts', {
                query_embedding: embedData.embedding,
                alpha_weight: 0.7,
                match_count: 5
            });
            const semTime = performance.now() - startSem;

            // C. Keyword Search (Lexical Baseline)
            const startKey = performance.now();
            const { data: keyData } = await supabase.rpc('keyword_search_prompts', {
                query_text: word,
                match_count: 5
            });
            const keyTime = performance.now() - startKey;

            finalResults.push({
                keyword: word,
                found: {
                    semantic: semData?.length || 0,
                    keyword: keyData?.length || 0
                },
                performance: {
                    semantic_db_ms: parseFloat(semTime.toFixed(2)),
                    keyword_db_ms: parseFloat(keyTime.toFixed(2))
                },
                cache: {
                    is_hit: embedData.cacheHit,
                    api_time_ms: parseFloat(embedData.time.toFixed(2))
                }
            });
            console.log(`| Sem: ${semData?.length || 0} | Key: ${keyData?.length || 0} | Done.`);
        } catch (e) {
            console.error(`| Error: ${e.message}`);
        }
    }

    fs.writeFileSync('./keyword_benchmark_results.json', JSON.stringify(finalResults, null, 2));
    console.log("\n📊 BENCHMARK COMPLETE: Results saved to 'keyword_benchmark_results.json'");
}

runBenchmark().catch(console.error);