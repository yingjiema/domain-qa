from fastapi import FastAPI

# from datasets import load_dataset
# from generative_pseudo_label_generator import GenerativePseudoLabelGenerator
# from sentence_transformers import SentenceTransformer, util

# corpus = ['Flu and COVID are both respiratory illnesses that arise from the spread of different viruses. The flu usually peaks between December and February, according to the U.S. Centers for Disease Control and Prevention (CDC), and the severity of the flu depends on the season and how the flu vaccine correlates to what circulates, says Dr. Preeti Malani, an infectious disease physician at the University of Michigan.',
#          'COVID-19 isn’t tied to a particular season although experts speculate that there will be another COVID wave between October and January.',
#          'Although having either the flu or COVID can present as asymptomatic, those with the flu typically feel symptoms right away, between one and four days after infection, while those with COVID can feel them from two to five days after or anywhere up to 14 days, per the CDC.',
#          'According to the CDC, the symptoms of any viral infection are similar, although it is more common to have a loss of smell and taste if you’re expiring a COVID-19 reaction versus the flu or a cold. Here are the symptoms, with information from the AAFA and the CDC.'
#          ]

app = FastAPI()

# @app.post('/retriever-train', tags=['Generative Pseudo Labeling'])
# async def main(corpus, retriever_dir='retriever'):
#     gpl = GenerativePseudoLabelGenerator()
#     gpl.add_documents(corpus)
#     gpl.create_embedding_retriever()
#     gpl.train()
#     gpl.save(retriever_dir)

@app.get('/', tags=['Health Check'])
async def root():
    return {'message': 'Generative Pseudo Labeler service is up!'}
