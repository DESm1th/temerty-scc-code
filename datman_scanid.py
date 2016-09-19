"""
Represents scan identifiers that conform to the TIGRLab naming scheme

Taken from TIGRLab/datman
"""
import os.path
import re

SCANID_RE = '(?P<study>[^_]+)_' \
            '(?P<site>[^_]+)_' \
            '(?P<subject>[^_]+)_' \
            '(?P<timepoint>[^_]+)_' \
            '(?P<session>[^_]+)'

SCANID_PHA_RE = '(?P<study>[^_]+)_' \
                '(?P<site>[^_]+)_' \
                '(?P<subject>PHA_[^_]+)' \
                '(?P<timepoint>)(?P<session>)'  # empty

FILENAME_RE = SCANID_RE + '_' + \
              r'(?P<tag>[^_]+)_' + \
              r'(?P<series>\d+)_' + \
              r'(?P<description>[^\.]*)' + \
              r'(?P<ext>\..*)?'

FILENAME_PHA_RE = SCANID_PHA_RE + '_' + \
              r'(?P<tag>[^_]+)_' + \
              r'(?P<series>\d+)_' + \
              r'(?P<description>[^\.]*)' + \
              r'(?P<ext>\..*)?'

SCANID_PATTERN       = re.compile('^'+SCANID_RE+'$')
SCANID_PHA_PATTERN   = re.compile('^'+SCANID_PHA_RE+'$')
FILENAME_PATTERN     = re.compile('^'+FILENAME_RE+'$')
FILENAME_PHA_PATTERN = re.compile('^'+FILENAME_PHA_RE+'$')

class ParseException(Exception):
    pass

class Identifier:
    def __init__(self, study, site, subject, timepoint, session):
        self.study = study
        self.site = site
        self.subject = subject
        self.timepoint = timepoint
        self.session = session

    def get_full_subjectid(self):
        return "_".join([self.study, self.site, self.subject])

    def get_full_subjectid_with_timepoint(self):
        ident = self.get_full_subjectid()
        if self.timepoint:
            ident += "_"+self.timepoint
        return ident

    def __str__(self):
        if self.timepoint:
            return "_".join([self.study,
                             self.site,
                             self.subject,
                             self.timepoint,
                             self.session])
        else:  # it's a phantom, so no timepoints
            return self.get_full_subjectid()

def parse(identifier):
    if type(identifier) is not str: raise ParseException()

    match = SCANID_PATTERN.match(identifier)
    if not match: match = SCANID_PHA_PATTERN.match(identifier)
    if not match: raise ParseException()

    ident = Identifier(study    = match.group("study"),
                       site     = match.group("site"),
                       subject  = match.group("subject"),
                       timepoint= match.group("timepoint"),
                       session  = match.group("session"))

    return ident

def parse_filename(path):
    fname = os.path.basename(path)
    match = FILENAME_PHA_PATTERN.match(fname)  # check PHA first
    if not match: match = FILENAME_PATTERN.match(fname)
    if not match: raise ParseException()

    ident = Identifier(study    = match.group("study"),
                       site     = match.group("site"),
                       subject  = match.group("subject"),
                       timepoint= match.group("timepoint"),
                       session  = match.group("session"))

    tag = match.group("tag")
    series = match.group("series")
    description = match.group("description")
    return ident, tag, series, description

def make_filename(ident, tag, series, description, ext = None):
    filename = "_".join([str(ident), tag, series, description])
    if ext:
        filename += ext
    return filename

def is_scanid(identifier):
    try:
        parse(identifier)
        return True
    except ParseException:
        return False

def is_phantom(identifier):
    try:
        x = parse(identifier)
        return x.subject[0:3] == 'PHA'
    except ParseException:
        return False



# vim: ts=4 sw=4:
