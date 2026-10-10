from episol import epipy
"""
this runs 3drism on a single methane molecule
in cyanide.
"""
def run_test():
    try:
        stage = "Initializing class and converting topology to .solute and generating idc-enabled topology"
        t = epipy('methane.gro','methane.top',convert=True,gen_idc=True)
        stage = "setting solvent"
        t.solvent("cyanide.sol")
        print("using solvent",t.solvent_top)
        stage = "setting resolution"
        #print("solvent path:",t.solute_top_path)
        t.rism(resolution=0.5)
        stage = "running rism on 1 thread"
        t.kernel()
        ###
        stage = "extracting g(r) data"
        g_r = t.select_grid('guv')
        print("resulting shape:",g_r.shape)
    except Exception as exc:
        print("!!!!!!!!!!!!!!!!!!!!")
        print("failed at:\n")
        print(stage)
        raise exc
#print("solvent path:",t.solute_top_path)
if __name__ == "__main__":
    run_test()
