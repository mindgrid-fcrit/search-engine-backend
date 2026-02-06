import fs from 'fs';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import crypto from 'crypto';
import 'dotenv/config';

// 1. SETUP
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: "text-embedding-004" });

const TEST_QUERIES = [
    "How to optimize Python code?", 
    "Help me write a professional email",
    "Creative story about a lost king", 
    "Excel formulas for complex math",
    "Debugging javascript memory leaks", 
    "Career advice for junior developers",
    "Latex code for tables", 
    "Marketing strategy for software", 
    "Interpret a dream about flying", 
    "Side scrolling game logic"
];

// 2. MATH HELPERS
function cosineSimilarity(vecA, vecB) {
    if (!Array.isArray(vecA) || !Array.isArray(vecB)) return 0;
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * (vecB[i] || 0), 0);
    const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    if (magA === 0 || magB === 0) return 0;
    return dotProduct / (magA * magB);
}

// 3. VECTOR & CACHE LOGIC
async function getVectorWithCache(text) {
    const hash = crypto.createHash('md5').update(text.trim().toLowerCase()).digest('hex');
    const startCache = performance.now();
    
    // Check Cache table
    const { data: cached } = await supabase.from('search_cache').select('embedding').eq('id', hash).single();
    
    if (cached) {
        // Ensure cached embedding is an array (parse if it's a string)
        const vector = typeof cached.embedding === 'string' ? JSON.parse(cached.embedding) : cached.embedding;
        return { embedding: vector, cacheHit: true, time: performance.now() - startCache };
    }

    // Gemini API Call
    const result = await embedModel.embedContent(text);
    const embedding = result.embedding.values;
    
    // Save to Cache
    await supabase.from('search_cache').upsert({ id: hash, query_text: text, embedding });
    return { embedding, cacheHit: false, time: performance.now() - startCache };
}

// 4. MAIN TEST RUNNER
async function runTest() {
    console.log("🔥 WARMING UP: Priming DB and Server connection...");
    // Warmup: perform one throwaway search to wake up the DB and Gemini
    const warmupVec = await getVectorWithCache("warmup");
    await supabase.rpc('hybrid_search_prompts', { query_embedding: warmupVec.embedding, match_count: 1 });
    console.log("✅ Warm-up complete.\n");

    let finalResults = [];

    for (const queryText of TEST_QUERIES) {
        process.stdout.write(`🔎 Testing: "${queryText}"... `);

        // A. GET EMBEDDING (Handles Cache vs API)
        const embedData = await getVectorWithCache(queryText);

        // B. DB-SIDE ARCHITECTURE (The Optimized Way)
        const startDB = performance.now();
        const { data: dbData, error: dbErr } = await supabase.rpc('hybrid_search_prompts', {
            query_embedding: embedData.embedding,
            alpha_weight: 0.7,
            match_count: 5
        });
        const dbTime = performance.now() - startDB;

        // C. SERVER-SIDE ARCHITECTURE (The Inefficient Way)
        const startServer = performance.now();
        const { data: allPrompts } = await supabase.from('prompts').select('id, content, embedding, votes, quality_score');
        
        const serverResults = allPrompts.map(p => {
            // CRITICAL FIX: Parse the vector string from Postgres if needed
            const vectorArray = typeof p.embedding === 'string' ? JSON.parse(p.embedding) : p.embedding;
            
            const sem = cosineSimilarity(embedData.embedding, vectorArray);
            const meta = (p.votes + p.quality_score) / 200.0;
            const rank = (sem * 0.7) + (meta * 0.3);
            return { id: p.id, rank };
        })
        .sort((a, b) => b.rank - a.rank)
        .slice(0, 5);
        
        const serverTime = performance.now() - startServer;

        finalResults.push({
            query: queryText,
            cache: {
                is_hit: embedData.cacheHit,
                fetch_time_ms: parseFloat(embedData.time.toFixed(2))
            },
            performance: {
                db_side_latency_ms: parseFloat(dbTime.toFixed(2)),
                server_side_latency_ms: parseFloat(serverTime.toFixed(2)),
                speed_improvement_factor: parseFloat((serverTime / dbTime).toFixed(2)) + "x"
            },
            data_transfer: {
                rows_fetched_from_db: allPrompts.length,
                results_returned_to_user: 5
            }
        });
        console.log("Success.");
    }

    fs.writeFileSync('./computational_latency_results.json', JSON.stringify(finalResults, null, 2));
    console.log("\n📊 TEST COMPLETE: Results saved to 'computational_latency_results.json'");
}

runTest().catch(console.error);