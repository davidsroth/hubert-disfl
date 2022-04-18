Code adapted from https://github.com/nihal111/NXT-Switchboard-Disfluency-Parser. 

## Parsing methodology:
Conversations are in form of text files with highligted with speaker A and B. We have disfluency annotations in xml files. The parser code parses the text files and tokenizes each sequence of text. Extract code annotates the files with disfluency and repairs and saves text files. For example, the extract code outputs <em>"Uh SILENCE first SILENCE um SILENCE I need TRACE to know SILENCE uh SILENCE how do you feel TRACE <e> about SILENCE <ip> uh SILENCE about <r> TRACE sending SILENCE uh SILENCE an elderly SILENCE uh SILENCE family member to a nursing home"</em> for the input <em>"okay uh first um i need to know uh h[ow]- how do you feel about uh about sending um an elderly uh family member to a nursing home"</em>. Here, the edit start is shown by `<e>`, interruption point by `<ip>` and end of repair by `<r>`.


To parse all the content of a file, usage like

```python parsing_all.py <filename, like sw2005> > <output_file_name>```

For example,

```python parsing_all.py sw2005 > result/sw2005```


To extract disfluency:

```python extract_all.py <filename, like sw2005> > <extracted_output_file_name>```

For example,

```python extract_all.py sw2005 > result_extracted/sw2005```
