# Render with: vhs demo/demo.tape   (https://github.com/charmbracelet/vhs)
Output demo/demo.gif

Set Shell "bash"
Set FontSize 20
Set Width 1240
Set Height 660
Set Padding 22
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set TypingSpeed 55ms

Hide
Type `export PATH="$PWD/.venv/bin:$PATH" && clear` Enter
Sleep 1s
Show

Type "# measure a RAG config on bundled synthetic data -- no keys, fully offline"
Enter
Sleep 900ms
Type "rag-eval bench samples/parts_sample.csv --skip-generation --max-qa 60"
Enter
Sleep 13s
Type "# retrieval scored against a ground-truth QA set. swap in your corpus and go."
Enter
Sleep 3s
