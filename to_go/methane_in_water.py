from episol import epipy
"""
this runs a 3drism and water placement test
on a single methane molecule
"""
def run_test():
    try:
        stage = "Initializing class and converting topology to .solute and generating idc-enabled topology"
        t = epipy('methane.gro','methane.top',convert=True,gen_idc=True)
        stage = "setting resolution"
        #print("solvent path:",t.solute_top_path)
        t.rism(resolution=0.5)
        stage = "running rism on 1 thread"
        t.kernel()
        ###
        stage = "extracting g(r) data"
        g_r = t.select_grid('guv')
        print("resulting shape:",g_r.shape)
        w = t.placement(10)
        print("placed",len(w),"waters")
    except Exception as exc:
        print("!!!!!!!!!!!!!!!!!!!!")
        print("failed at:\n")
        print(stage)
        raise exc
#print("solvent path:",t.solute_top_path)
if __name__ == "__main__":
    run_test()
