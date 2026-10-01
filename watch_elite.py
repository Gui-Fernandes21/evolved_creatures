import sys
import time

import pybullet as p

import creature
import genome
from mountain_simulation import MountainSimulation


def main(csv_file):
    sim = MountainSimulation(gui=True)

    cr = creature.Creature(gene_count=1)
    dna = genome.Genome.from_csv(csv_file)
    cr.update_dna(dna)
    
    # set the camera to a good position for watching the creature
    p.resetDebugVisualizerCamera(
        cameraDistance=15,  
        cameraYaw=45,  
        cameraPitch=-35,  
        cameraTargetPosition=[0, 0, 2],  
    )

    # Run for 30 wall-clock seconds at real time
    total_steps = 240 * 30
    xml_file = "watch.urdf"
    with open(xml_file, "w") as f:
        f.write(cr.to_xml())
    cid = p.loadURDF(xml_file, physicsClientId=sim.pid)
    p.resetBasePositionAndOrientation(cid, list(sim.spawn_position),
                                      [0, 0, 0, 1], physicsClientId=sim.pid)


    best_z = -999
    for step in range(total_steps):
        p.stepSimulation(physicsClientId=sim.pid)
        if step % 24 == 0:
            sim.update_motors(cid, cr)
        pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=sim.pid)
        best_z = max(best_z, pos[2])
        time.sleep(1.0 / 240)

    print("highest z reached:", round(best_z, 3),
          "summit z:", round(sim.summit[2], 3))


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: python watch_elite.py <elite_genome.csv>"
    main(sys.argv[1])
