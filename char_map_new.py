"""
Defines two dictionaries for converting 
between text and integer sequences.
"""

char_map_str = """
ঀ 0
ঁ 1
ং 2
ঃ 3
অ 4
আ 5
ই 6
ঈ 7
উ 8
ঊ 9
ঋ 10
ঌ 11
এ 12
ঐ 13
ও 14
ঔ 15
ক 16
খ 17
গ 18
ঘ 19
ঙ 20
চ 21
ছ 22
জ 23
ঝ 24
ঞ 25
ট 26
ঠ 27
ড 28
ঢ 29
ণ 30
ত 31
থ 32
দ 33
ধ 34
ন 35
প 36
ফ 37
ব 38
ভ 39
ম 40
য 41
র 42
ল 43
শ 44
ষ 45
স 46
হ 47
় 48
ঽ 49
া 50
ি 51
ী 52
ু 53
ূ 54
ৃ 55
ৄ 56
ে 57
ৈ 58
ো 59
ৌ 60
্ 61
ৎ 62
ৗ 63
ড় 64
ঢ় 65
য় 66
ৠ 67
০ 68
১ 69
২ 70
৩ 71
৪ 72
৫ 73
৬ 74
৭ 75
৮ 76
৯ 77
ৱ 78
৲ 79
৴ 80
<SPACE> 81
"""
# the "blank" character is mapped to 84

char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index) + 1] = ch
index_map[82] = ' '
'''
char_map['\u200c'] = 133
index_map[134] = '\u200c'
char_map['\x93'] = 134
index_map[135] = '\x93'
char_map['\x94'] = 135
index_map[136] = '\x94'
char_map['\u200d'] = 136
index_map[137] = '\u200d'
char_map['…'] = 137
index_map[138] = '…'
'''
