# from topik import read_input, tokenize, vectorize, run_model, visualize
# raw_data = read_input("trump.json")
# content_field = 'TRUMP:'
# for item in raw_data:
#     print item
#
# raw_data = ((hash(item[content_field]), item[content_field]) for item in raw_data)
# tokenized_corpus = tokenize(raw_data)
# vectorized_corpus = vectorize(tokenized_corpus)
# ntopics = 7
# model = run_model(vectorized_corpus, ntopics=ntopics)
# plot = visualize(model)


from topik.run import run_model

run_model('trump.json', field='TRUMP:', model='lda_online', r_ldavis=True, output_file=True)
