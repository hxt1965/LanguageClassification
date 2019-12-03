en = []
nl = []
with open('data.txt', 'r') as fp:
    for line in fp:
        #print(line)
        s = line.split('|')
        if s[0] == 'en':
            en.append(s[1])
        else:
            nl.append(s[1])
        
words_long_en = 0
words_short_en = 0
words_long_nl = 0
words_short_nl = 0
vowel_cnt_en = 0
vowel_cnt_nl = 0

print(en[200])
print(nl[200])

vowels = ['a', 'e', 'i', 'o', 'u']


for line in en:
    for word in line:
        for c in word:
            if c in vowels:
                vowel_cnt_en = vowel_cnt_en + 1 
        if len(word) >= 7 :
            words_long_en = words_long_en + 1
        elif len(word) <=4:
            words_short_en = words_short_en + 1


for line in nl:
    for word in line:
        for c in word:
            if c in vowels:
                vowel_cnt_nl = vowel_cnt_nl + 1
        if len(word) >=7 :
            words_long_nl = words_long_nl + 1
        elif len(word) <=4:
            words_short_nl = words_short_nl + 1

print('long words per line in en: ', words_long_en)
print('short words per line in en: ', (words_short_en/len(en)))
print('long words per line in nl: ', words_long_nl)
print('short words per line in nl: ', (words_short_nl/len(nl)))
print('vowels per line in en: ', (vowel_cnt_en / len(en)))
print('vowels per line in nl: ', (vowel_cnt_nl / len(nl)))