### Web Access using Python

#### Regular Expressions

Powerful and cryptic.

In computing, a regular expression, also referred to as "regex" or "regexp", provides a concise and felxible means for matching strings of text, such as particular characters, words, or patterns of characters. 

A regular expression is written in a formal language that can be interpreted by a regular expression processor.

- Lots of operating systems have used regular expressions as a more intelligent form of search. 


Python Regular Expression Quick Guide

        ^        Matches the beginning of a line
        $        Matches the end of the line
        .        Matches any character
        \s       Matches whitespace
        \S       Matches any non-whitespace character
        *        Repeats a character zero or more times
        *?       Repeats a character zero or more times 
                 (non-greedy)
        +        Repeats a character one or more times
        +?       Repeats a character one or more times 
                 (non-greedy)
        [aeiou]  Matches a single character in the listed set
        [^XYZ]   Matches a single character not in the listed set
        [a-z0-9] The set of characters can include a range
        (        Indicates where string extraction is to start
        )        Indicates where string extraction is to end


The Regular Expression Module

```python

import re  # for using regular expressions

# To see if a string matches a regular expression
re.search()
# Similar to using 
find()
# method for string


#Extracting portions of a string that match with a certain regular #expression can be done using
re.findall()


## Some equivalent functions:

find('some') ~  re.search('some')
startwith('some') ~ re.search('^some')
## Please note of the special characters from above.


```

Example of Wild-Card Characters:

```python

# Consider the following regular expression:

^X.*:

# The dot character match any characters.
# The asterisk character, the character is "any number of times"

```

#### Fine Tuning our Match

Depending on the pupose we can Fine Tune our match depending on how "clean" our data is. We may want to narrow match down a bit.

```python

^X-\S+:

# \S signifies the non-blank character that must be repeated one or more time followed by colon.

## Example statements that will match:

s = 'X-Miracle-: '

# The following example is not a match because there;s a whitespace character between
# first character and the colon.
not_a_match = 'X-Plane is behing schedule: two weeks'


```
