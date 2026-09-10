# Put your plate photos here

Anything in this folder is ignored by git, so your images stay local:

    python -m cellcounter count images/ --out-dir results

JPEG, PNG and TIFF work directly. iPhone `.HEIC` files need
`pip install pillow-heif`, or export them as JPEG first.
