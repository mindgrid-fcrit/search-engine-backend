import 'dotenv/config';
import express from 'express';
import crypto from 'crypto';
import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

// Clients
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "text-embedding-004" });

/**
 * CORE HELPER: getVector
 * @param {string} text - The text to vectorize
 * @param {boolean} useCache - If true, checks/saves to search_cache (for searches)
 */
async function getVector(text, useCache = false) {
  const cleanText = text.trim().toLowerCase();
  const hash = crypto.createHash('md5').update(cleanText).digest('hex');

  // 1. Optional Cache Check
  if (useCache) {
    const { data: cached } = await supabase
      .from('search_cache')
      .select('embedding')
      .eq('id', hash)
      .single();
    if (cached) return cached.embedding;
  }

  // 2. Generate Embedding
  const result = await model.embedContent({
    content: { parts: [{ text: cleanText }] },
    outputDimensionality: 768,
  });
  const embedding = result.embedding.values;

  // 3. Optional Cache Store
  if (useCache) {
    await supabase.from('search_cache').upsert({
      id: hash,
      query_text: cleanText,
      embedding: embedding
    });
  }

  return embedding;
}

/**
 * POST /api/prompts
 * Adds a permanent prompt. Does NOT pollute the search cache.
 */
app.post('/api/prompts', async (req, res) => {
  try {
    const { text, category, votes, quality_score } = req.body;
    
    // We pass 'false' because we don't need this in the search_cache
    const embedding = await getVector(text, false);

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
    res.status(201).json({ message: "Prompt stored permanently", data: data[0] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/search
 * Performs hybrid search. USES search_cache for speed and API savings.
 */
app.get('/api/search', async (req, res) => {
  try {
    const { q, alpha = 0.5 } = req.query;
    if (!q) return res.status(400).json({ error: "Query 'q' is required" });

    // We pass 'true' to enable the cache layer
    const queryVector = await getVector(q, true);

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

// Initialize the Text model for description and execution
const textModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

/**
 * GET /api/prompts
 * Fetches ALL prompts from the database for the list view.
 * Privacy: Does NOT return 'content' or 'embedding'.
 */
app.get('/api/prompts', async (req, res) => {
  try {
    // 1. Fetch only safe columns from Supabase
    const { data, error } = await supabase
      .from('prompts')
      .select('id, description, category, votes, quality_score, created_at')
      .order('created_at', { ascending: false }); // Newest first

    if (error) throw error;

    // 2. Return the list (User sees description, but prompt is hidden)
    res.status(200).json({
      count: data.length,
      prompts: data
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/**
 * 2. EXECUTE PROMPT (The "Black Box")
 * User sends data -> Server joins it with secret prompt -> User gets AI result.
 */
app.post('/api/prompts/:id/execute', async (req, res) => {
  try {
    const { id } = req.params;
    const { userInput } = req.body; // The data the user wants processed

    if (!userInput) return res.status(400).json({ error: "User input is required" });

    // Fetch the secret prompt content from DB
    const { data: prompt, error } = await supabase
      .from('prompts')
      .select('content')
      .eq('id', id)
      .single();

    if (error || !prompt) return res.status(404).json({ error: "Prompt not found" });

    // Combine your secret prompt with the user's data
    const finalInstruction = `
      System Instructions: ${prompt.content}
      
      User Data to process: ${userInput}
      
      Please provide the result based on the instructions above.
    `;

    // Execute via Gemini
    const result = await textModel.generateContent(finalInstruction);
    const response = await result.response;
    
    // Return only the text result
    res.status(200).json({
      success: true,
      output: response.text()
    });

  } catch (err) {
    res.status(500).json({ error: "Execution failed: " + err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🚀 Engine live at port ${PORT}`));