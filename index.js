import 'dotenv/config';
import express from 'express';
import { createClient } from '@supabase/supabase-js';
import { HfInference } from '@huggingface/inference';

const app = express();
app.use(express.json()); // Essential for parsing JSON bodies

// Initialize Clients
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const hf = new HfInference(process.env.HF_TOKEN);

/** * Helper: Vectorize text using HF 
 */
async function getVector(text) {
  return await hf.featureExtraction({
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    inputs: text,
  });
}

/** * Endpoint: Add a new prompt (Auto-vectorized)
 */
app.post('/api/prompts', async (req, res) => {
  try {
    // Remove 'id' from the request body destructuring
    const { text, category, votes, quality_score } = req.body;

    const embedding = await getVector(text);

    const { data, error } = await supabase
      .from('prompts')
      .insert([{ 
        content: text, // No ID sent here!
        category, 
        votes: votes || 0, 
        quality_score: quality_score || 0, 
        embedding 
      }])
      .select(); // This will return the auto-generated ID

    if (error) throw error;
    res.status(201).json({ message: "✅ Prompt added", data: data[0] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/** * Endpoint: Hybrid Search
 */
app.get('/api/search', async (req, res) => {
  try {
    const { q, alpha = 0.5 } = req.query;

    if (!q) return res.status(400).json({ error: "Query parameter 'q' is required" });

    // 1. Vectorize the search query
    const queryVector = await getVector(q);

    // 2. Call Supabase RPC
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
app.listen(PORT, () => console.log(`Engine running on http://localhost:${PORT}`));