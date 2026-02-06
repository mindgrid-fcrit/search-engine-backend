import { createClient } from '@supabase/supabase-js';
import { GoogleGenerativeAI } from '@google/generative-ai';
import 'dotenv/config';

// 1. Setup
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embedModel = genAI.getGenerativeModel({ model: "gemini-embedding-001" });

async function reindexPrompts() {
    console.log("🔄 Starting Re-indexing process with gemini-embedding-001...");

    // 2. Fetch all prompts
    const { data: prompts, error } = await supabase
        .from('prompts')
        .select('id, content');

    if (error) {
        console.error("❌ Error fetching prompts:", error);
        return;
    }

    console.log(`Found ${prompts.length} prompts to update.`);

    for (let i = 0; i < prompts.length; i++) {
        const { id, content } = prompts[i];

        try {
            process.stdout.write(`Processing [${i + 1}/${prompts.length}]... `);

            // 3. Generate new vector
            // We force 768 dimensions to fit your existing DB schema
            const result = await embedModel.embedContent({
                content: { parts: [{ text: content }] },
                taskType: "RETRIEVAL_DOCUMENT", 
                outputDimensionality: 768, 
            });
            const newEmbedding = result.embedding.values;

            // 4. Update the row
            const { error: updateError } = await supabase
                .from('prompts')
                .update({ embedding: newEmbedding })
                .eq('id', id);

            if (updateError) throw updateError;

            console.log("✅ Updated");

            // Avoid rate limits (Sleep for 200ms between rows)
            await new Promise(r => setTimeout(r, 200));

        } catch (err) {
            console.error(`\n❌ Failed at ID ${id}:`, err.message);
        }
    }

    console.log("\n✨ Re-indexing complete!");
}

reindexPrompts().catch(console.error);