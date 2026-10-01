import unittest
import creature

class TestCreature(unittest.TestCase):
    def test_creature_genome(self):
        cr = creature.Creature(gene_count=10)
        print(cr.get_flat_links())
        self.assertEqual(len(cr.dna), 10)

        
unittest.main()