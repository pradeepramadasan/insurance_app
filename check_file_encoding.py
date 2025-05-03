import re

path = r"c:\Users\pramadasan\insurance_app\workflow\process.py"
data = open(path, "rb").read()

nul_count = data.count(b"\x00")
crlf_count = len(re.findall(rb"\r\n", data))
# “LF” not preceded by CR:
lf_count   = len(re.findall(rb"(?<!\r)\n", data))
# “CR” not followed by LF:
cr_count   = len(re.findall(rb"\r(?!\n)", data))

print(f"Null bytes (00): {nul_count}")
print(f"CRLF sequences (0D0A): {crlf_count}")
print(f"Orphan LF      (0A not preceded by 0D): {lf_count}")
print(f"Orphan CR      (0D not followed  by 0A): {cr_count}")