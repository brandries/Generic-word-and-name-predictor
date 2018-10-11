# This is a Python 3.6 files
# Containing classes needed for text processing

class WordLength:
    def __init__(self):
        pass
    
    def transform(self, word):
        return len(word)


class HasSpaces:
    def __init__(self):
        pass
    
    def transform(self, word):
        strp = word.split(' ')
        if len(strp) > 1:
            return 1
        else:
            return 0
    
class HasNumbers:
    def __init__(self):
        pass
    
    def transform(self, word):
        sets = 0
        for i in word:
            if i.isdigit() == False:
                sets = 0
            elif i.isdigit() == True:
                sets = 1
                break
        return sets


class NumberOfNumbers:
    def __init__(self):
        pass
    
    def transform(self, word):
        sets = 0
        for i in word:
            if i.isdigit() == False:
                pass
            elif i.isdigit() == True:
                sets += 1
        return sets


class HasUppers:
    def __init__(self):
        pass

    def transform(self, word):
        sets = 0
        for i in word:
            if i.isupper() == False:
                sets = 0
            elif i.isupper() == True:
                sets = 1
                break
        return sets


class NumberOfUppers:
    def __init__(self):
        pass

    def transform(self, word):
        sets = 0
        for i in word:
            if i.isupper() == False:
                pass
            elif i.isupper() == True:
                sets += 1
        return sets

class Vowels:
    def __init__(self):
        pass

    def transform(self, word):
        vow = 0
        for i in word:
            if i.lower() in ['a', 'e', 'i', 'u', 'o']:
                vow += 1
        return vow/len(word)


class Punctuation:
    def __init__(self):
        pass

    def transform(self, word):
        import string
        punc = 0
        for i in word:
            if i.lower() in list(string.punctuation):
                punc += 1
        return punc


class MoreCapitals:
    def __init__(self):
        pass

    def transform(self, word):
        sets = 0
        for i in word:
            if i.isupper() == False:
                pass
            elif i.isupper() == True:
                sets += 1

        if sets > 1:
            strp = word.split(' ')
            if len(strp) == 1:
                return 0
            else:
                return 1
        else:
            return 1    
    

class Syllables:
    def __init__(self):
        pass

    def transform(self, word):
        from textstat.textstat import textstat
        return textstat.syllable_count(word)


class Readability:
    def __init__(self):
        pass

    def transform(self, word):
        from textstat.textstat import textstat
        return textstat.automated_readability_index(word)
