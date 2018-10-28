# clean.py
```
python clean.py inputFile | awk '!uniq[substr($0, 0, 10000)]++' > outFile
```
- Basic cleaning of the `inputFile` and writing to `outFile`, each line in `outFile` being a sentence from the `inputFile`.
- May have errors, ignore them or manually correct if you want.

# script.py

```
python script.py inputFile outFile annotationFile
```
- Send the `inputFile` (output of `clean.py`).
- Each line from the `inputFile` pops up and you will be asked to tag it.
- The details of the key for each tag will be output under each sentence.
- Enter the corresponding tag and that line will be appended to `outFile` and your tag to the `annotationFile`.
- Used RECHECK(R) tag incase you are not sure about it. Do it manually later, etc.
- Used IGNORE(I) tag incase that sentence is trash and is not cleaned well from `clean.py` file.
- The line and the tag wont be written incase of IGNORE tag.
