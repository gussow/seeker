from unittest import TestCase

import seeker

TEST_INPUT = {
    "bacterial": ("example_input/BACE2.txt", "example_input/BACE2.output.txt"),
    "phage": ("example_input/PGE.txt", "example_input/PGE.output.txt")
}


class TestSeeker(TestCase):
    longMessage = True

    def test_fragment(self):
        for name, sample_data in TEST_INPUT.items():
            sample_input_path, sample_output_path = sample_data
            sample_output = open(sample_output_path).read().strip()

            seeker_fasta = seeker.SeekerFasta(sample_input_path)
            output = "\n".join(seeker_fasta.phage_or_bacteria())
            self.assertEqual(output, sample_output, msg="Unequal output for {}".format(name))