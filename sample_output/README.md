# Sample output

No sample JSON is pre-included here — every evaluation makes a real LLM
call (see the main README's Setup section for OpenRouter/Anthropic/OpenAI),
so a bundled sample would either need a real API key to generate honestly,
or be fake data that misrepresents what the system actually does.

Generate a genuine sample for BOTH engines here with one command, once you
have a key set:

```bash
export OPENROUTER_API_KEY="sk-or-..."   # or ANTHROPIC_API_KEY / OPENAI_API_KEY
python -c "
import json, os
from database.db_setup import init_db, get_active_criteria
from agents.orchestrator import run_batch_evaluation
from agents_langgraph.langgraph_pipeline import run_langgraph_batch_evaluation

init_db()
criteria = get_active_criteria()
pdf_dir = 'sample_data/supplier_pdfs'
dates = {'Apex_Systems.pdf':'2026-02-10','BrightPath_Tech.pdf':'2026-02-05','NexaWorks.pdf':'2026-02-08','Orbit_Digital.pdf':'2026-02-12'}
exp = {'Apex_Systems.pdf':8.0,'BrightPath_Tech.pdf':3.0,'NexaWorks.pdf':7.5,'Orbit_Digital.pdf':9.0}
supplier_inputs = []
for fname in sorted(os.listdir(pdf_dir)):
    with open(os.path.join(pdf_dir, fname), 'rb') as f:
        pdf_bytes = f.read()
    name = fname.rsplit('.',1)[0].replace('_',' ')
    supplier_inputs.append({'supplier_name': name, 'submission_date': dates[fname], 'experience_rating': exp[fname], 'pdf_bytes': pdf_bytes})

direct_result = run_batch_evaluation(supplier_inputs, criteria)
with open('sample_output/sample_run_result_direct.json', 'w') as f:
    json.dump(direct_result, f, indent=2, default=str)
print('Wrote sample_output/sample_run_result_direct.json, RFP_RUN_ID:', direct_result['rfp_run_id'])

langgraph_result = run_langgraph_batch_evaluation(supplier_inputs, criteria)
with open('sample_output/sample_run_result_langgraph.json', 'w') as f:
    json.dump(langgraph_result, f, indent=2, default=str)
print('Wrote sample_output/sample_run_result_langgraph.json, RFP_RUN_ID:', langgraph_result['rfp_run_id'])
"
```

Note: the two engines' rankings are not expected to match exactly — each
makes its own independent LLM call, and two separate calls to the same
model aren't guaranteed to return identical scores. Both are legitimate,
independent "completed run" samples in their own right.

The same steps exist in `notebooks/VendorScope_Colab.ipynb` (Step 9, the
JSON export step) if you'd rather run it there.
