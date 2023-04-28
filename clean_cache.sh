#remove cache
rm -fr manapy/*/__pyc*
rm -fr manapy/__pyc*
rm -fr manapy/*/*/__pyc*

#remove meshes and results
rm -fr tests/meshes*
rm -fr tests/res*
rm -fr manapy/examples/res*
rm -fr manapy/examples/meshes*

