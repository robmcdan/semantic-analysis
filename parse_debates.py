import csv
import re
import os
from glob import glob

class DebateParser:

    def __init__(self, path):
        self.debates = self.__build_debate_list(path)
        self.statements = dict()
        print ""

    def __build_debate_list(self, path):
        return [y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.txt'))]

    def parse(self):
        for debate in self.debates:
            self.statements.update(self.__split_transcript_by_speaker(debate))

    def build_text_for_mallet(self, output_file_path):
        with open(output_file_path, 'wb+') as output_file:
            for speaker in self.statements.keys():
                for statement in self.statements[speaker]:
                    text = statement[0]
                    dem = statement[1]
                    gop = statement[2]
                    debate_number = statement[3]
                    output_file.write(text.replace('\n', ' ') + "\n")

    def get_statements_by_speaker(self, speaker, debate_number, csv_file="../resources/debates/gop/raw_text.tsv"):
        statements = list()
        with open(csv_file, 'rb') as open_file:
            reader = csv.reader(open_file, delimiter="\t")
            for row in reader:
                if int(row[2]) == debate_number and row[0] == speaker:
                    statements.append(row[1])

        return statements

    def __split_transcript_by_speaker(self, transcript_path):
        with open(transcript_path, 'rb') as transcript_file:
            text = transcript_file.read()

        statements = dict()
        # assuming there is only one digit in the filename, and that is the corresponding debate #
        debate_number = re.findall("\d+", transcript_path)[0]
        dem = gop = False;
        if "gop" in transcript_path.lower():
            gop = True
        elif "dem" in transcript_path.lower():
            dem = True
        assert dem != gop

        # debate transcript follows <SPEAKER:> convention before each statement
        speaker_regex = re.compile("(^[A-Z]+:\s*)", re.MULTILINE)
        result = speaker_regex.split(text)
        current_speaker = None
        for chunk in result:
            if chunk != '':
                if speaker_regex.match(chunk):
                    current_speaker = chunk.replace(" ", "").replace(":", "")
                    if current_speaker not in statements.keys():
                        statements[current_speaker] = list()
                    continue
                elif current_speaker is not None:
                    statements[current_speaker].append([chunk, dem, gop, debate_number])
        return statements

    def save_to_tsv(self, output):
        with open(output, 'wb+') as output_file:
            writer = csv.writer(output_file, delimiter="\t")
            writer.writerow(["Speaker", "Statement", "Democrat", "GOP", "Debate Number"])
            for speaker in self.statements.keys():
                for statement in self.statements[speaker]:
                    text = statement[0]
                    dem = statement[1]
                    gop = statement[2]
                    debate_number = statement[3]
                    writer.writerow([speaker, text.replace('\n', ' '), dem, gop, debate_number])

if __name__ == "__main__":
    parser = DebateParser("./data/debates")
    parser.parse()
    parser.build_text_for_mallet('./data/mallet_raw_statements.txt')
    parser.save_to_tsv('./data/raw_statements.tsv')
