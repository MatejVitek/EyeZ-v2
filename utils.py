from collections import defaultdict
import getpass
import os
import platform
import random
import re
import socket


UNIX = defaultdict(lambda: ('/hdd', os.path.join('/media', getpass.getuser(), 'All Your Base')))
WINDOWS = defaultdict(lambda: ('C:', 'D:', 'E:', 'F:', 'G:', 'H:'))
# WINDOWS['Funny-Computer-Name'] = ('F:', 'G:', 'D:', 'E:')


def multi_replace(string, replacements, ignore_case=False):
	if ignore_case:
		replacements = dict((k.lower(), v) for (k, v) in replacements.items())
	rep = map(re.escape, sorted(replacements, key=len, reverse=True))
	pattern = re.compile("|".join(rep), re.I if ignore_case else 0)
	return pattern.sub(lambda match: replacements[match.group(0)], string)


class _PartialFormatterDict(dict):
	def __missing__(self, key):
		return '{' + key + '}'


def partial_format(s, **kw):
	return s.format_map(_PartialFormatterDict(**kw))


def alphanum(s):
	return re.sub(r'[^0-9a-zA-Z]', '', s)


def get_eyez_dir():
	pc_name = socket.gethostname()
	for disk in (WINDOWS[pc_name] if platform.system().lower() == 'windows' else UNIX[pc_name]):
		eyez = os.path.join(disk, 'EyeZ')
		if os.path.isdir(eyez):
			return eyez


class Info:
	def __init__(self, gender, age, color):
		self.gender = gender
		self.age = int(age)
		self.color = color


def get_id_info(path=os.path.join(get_eyez_dir(), 'SBVPI', 'SBVPI_Gender_Age_Colour.txt')):
	try:
		with open(path, 'r') as f:
			return {
				(id + 1 + mirror_offset): Info(*line.split())
				for (id, line) in enumerate(f.readlines())
				for mirror_offset in (0, 55)
			}
	except IOError:
		return None


def shuffle(l):
	random.shuffle(l)
	return l
