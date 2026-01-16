import 'dotenv/config';
import express from 'express';
import crypto from 'crypto';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';

const app = express();
app.use(express.json());

// Initialize Supabase
const supabase = createClient(
  process.env.SUPABASE_URL, 
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "text-embedding-004" });

/**
 * HELPER: Get Vector with MD5 Caching
 * This checks Supabase first before calling Gemini.
 */
async function getVector(text) {
  const cleanText = text.trim().toLowerCase();
  
  // 1. Create MD5 Hash of the query
  const hash = crypto.createHash('md5').update(cleanText).digest('hex');

  // 2. Check Supabase Cache
  const { data: cached } = await supabase
    .from('search_cache')
    .select('embedding')
    .eq('id', hash)
    .single();

  if (cached) {
    console.log("🚀 Cache Hit: Using saved embedding");
    return cached.embedding;
  }

  // 3. Cache Miss: Call Gemini API
  console.log("☁️ Cache Miss: Generating new embedding via Gemini...");
  const result = await model.embedContent({
    content: { parts: [{ text: cleanText }] },
    outputDimensionality: 768, // Industry Standard
  });
  
  const embedding = result.embedding.values;

  // 4. Save to Cache (Permanent)
  // We use upsert to prevent errors if two people search the same thing at once
  await supabase.from('search_cache').upsert({
    id: hash,
    query_text: cleanText,
    embedding: embedding
  });

  return embedding;
}

/**
 * Endpoint: Add a new prompt
 */
app.post('/api/prompts', async (req, res) => {
  try {
    const { text, category, votes, quality_score } = req.body;
    const embedding = await getVector(text);

    const { data, error } = await supabase
      .from('prompts')
      .insert([{ 
        content: text, 
        category, 
        votes: votes || 0, 
        quality_score: quality_score || 0, 
        embedding 
      }])
      .select();

    if (error) throw error;
    res.status(201).json({ message: "✅ Prompt added", data: data[0] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * Endpoint: Hybrid Search (Now with Cache)
 */
app.get('/api/search', async (req, res) => {
  try {
    const { q, alpha = 0.5 } = req.query;
    if (!q) return res.status(400).json({ error: "Query 'q' is required" });

    // This will now check the cache automatically!
    const queryVector = await getVector(q);

    const { data, error } = await supabase.rpc('hybrid_search_prompts', {
      query_embedding: queryVector,
      alpha_weight: parseFloat(alpha),
      match_count: 5
    });

    if (error) throw error;
    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Engine running on port ${PORT}`));