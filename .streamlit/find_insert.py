with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# Find insertion point after OpenAI setup
for i, line in enumerate(lines):
    if 'Error in OpenAI call' in line:
        print(f'Found OpenAI section end at line {i}')
        break
        
print(f'Total lines in original: {len(lines)}')
