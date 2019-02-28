from bs4 import BeautifulSoup


file = './data/pdtsc_001.wdata'

with open(file, 'r', encoding='utf8') as f:
    raw = f.read()

soup = BeautifulSoup(raw, 'xml')

list_member_markups = soup.find_all(lambda tag: tag.name == 'LM' and tag.has_attr('id'))
start_time_markups = [LM.find('start_time') for LM in list_member_markups]
end_time_markups = [LM.find('end_time') for LM in list_member_markups]
token_markups = [LM.find_all('token') for LM in list_member_markups]

print([LM for LM in list_member_markups][2])
print(start_time_markups)
print(end_time_markups)
print([token.text.lower() for token in token_markups[5]])
print([token for token in token_markups if not token])
empty_idxs = [i for i, token in enumerate(token_markups) if not token]
print([start_time_markups[i] for i in empty_idxs])
print([end_time_markups[i] for i in empty_idxs])

print(token_markups[57])

token_markups.pop()

token_markups = [token for i, token in enumerate(token_markups) if i not in empty_idxs]
print(token_markups[0])
print([token for token in token_markups if not token])

# TODO: Create a class from all of this so that you can call data1 = Data(file)
# TODO: data1.start_times, data1.end_times, data1.tokens
# TODO: use one-hot encoding to encode the alphabet letters (graphemes) into vectors

# TODO: vypořádat se s počátečními LM, ve kterých nejsou tagy start_time a stop_time (if None --> ignore)
# TODO: jak nejlépe odstranit tagy start_time, end_time (regexp vs soup.text)
# TODO: jak nakrmit data do MFCC (uděláme z MFCC class?)

# start_times = [st.text for st in start_time_markups]
# print(start_times)
