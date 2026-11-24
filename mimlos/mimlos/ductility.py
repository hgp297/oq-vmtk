'''
Hazus reference tables from Chapter 5 of the MH 2.1
Technical Manual.
'''

import pandas as pd
from openquake.vmtk.units import units
import numpy as np
from mimlos.mimlos.nbcc import DUCTILITY_LOOKUP_TABLE, R_D_TABLE

SEG_TO_FEMA_TYPOLOGY = {
    "WLF-P9": "W1",
    "WLF": "W2",
    "WPB": "W2",
    "SMF": "S1",
    "SBF": "S2",
    "SLF": "S3",
    "SCW": "S4",
    "SIW": "S5",
    "CMF": "C1",
    "CSW": "C2",
    "CIW": "C3",
    "PCW": "PC1",
    "PCF1": "PC2",
    "PCF2": "PC2",
    "RML": "RM1",
    "RMC": "RM2",
    "URM": "URM",
    "CFS1": "W2",
    "CFS2": "S3",
}

def get_p695_ductility():
    '''
    use the following to get a 2025 value
    
    then using SEG-adjustment-per-era
    mu_year = (Rd_year/Rd_2025) * mu_2025

    default method is to use literature review to identify
    fema p-695-like study and reconstruct capacity curves
    using Omega (design-to-peak) and mu (yield-to-80%Vmax displacement)

    below is estimation for component-based ductility estimation

    generalized methodology for post-capping estimation
    for each LFRS, identify dominant mechanism/component
    go to Appendix of SEG and get all relevant components
    average out the d, e, and c values

    TODO: for e (post-cap residual plateau) consider taking the ratio
    of SEG e-to-d value, then applying that on top of the mu values
    (often collected from P-695 instead of SEG). This allows the two values 
    to be tied together relatively rather than absolutely
    '''


    ########## wood light frame

    # sourced from FEMA P-695 (ATC-63), FEMA P-2139-2
    # Goda & Atkinson (2011) for Canada
    # 1-5 stories, various wall aspect ratios

    # categorize commercial & multifamily as wlf, single family as wlf-p9
    wlf_mu = np.array([9.9, 7.1, 12.4, 7.9, 5.8, 5.4, 
                             7.0, 9.9, 7.7, 7.1, 6.2, 5.7, 8.0])
    
    wlfp9_mu = np.array([9.9, 9.6, 9.9, 7.7, 8.0, 0.018/0.0055, 0.016/0.0045, 0.012/0.004])
    wpb_mu = np.array([1.7, 7.3, 4.7, 3.5])

    wd_source = np.mean(wlf_mu)
    wlfp9_d_source = np.mean(wlfp9_mu)
    wpb_d_source = np.mean(wpb_mu)


    # post-cap: from SEG component
    # also sourced from FEMA P-2139-2 best estimate properties of simplified backbone curve

    # Table F.3, F.4
    # all wood shear wall components, h/b < 1.0
    
    we_source = np.mean(np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0, 8.0, 4.0])/0.75)

    # use ratio of SEG point e and d to bump P-695 ductility
    seg_wd_source = np.array([4.0, 2.6, 3.1, 2.3, 2.0, 3.6, 3.6, 3.5, 4.6, 5.0, 4.4, 5.7, 5.7, 4.4, 3.8, 5.0])
    seg_we_source = np.array([5.0, 3.6, 4.0, 3.0, 2.5, 4.0, 4.0, 4.0, 5.0, 6.0, 5.0, 6.3, 6.3, 5.0, 4.0, 9.0])
    wood_e_to_d_ratio = np.mean(np.divide(seg_we_source, seg_wd_source))
    
    wc_source = np.mean([0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

    ########## steel moment frame

    # these should  probably go into bilinear model, or extend out much more
    # sourced from FEMA P-695 studies 
    # (NIST TN 2084, NIST GCR 10-917-8, Fahnestock et al. (2020), Zareian et al. (2010), Elkady & Lignos (2014))
    # for NIST TN 2084 study, treat 4-story as low-rise
    # for Fahnestock, treat 8-story as mid-rise
    # mix of ELF and RSA design
    smfl_mu = np.array([3.76, 4.72, 5.3, 3.76, 4.02, 4.84, 5.4, 4.5,
                        3.86, 5.01, 2.58, 2.18, 3.86, 4.34, 2.37, 2.94])
    smfm_mu = np.array([1.09, 2.50, 2.17, 3.98, 2.74, 2.35, 1.91, 2.34, 4.1, 3.1,
                        4.60, 4.57, 4.56, 4.91,
                        5.59, 3.96, 4.98, 2.77,
                        3.67, 4.04, 3.34, 3.60, 2.88, 3.14, 2.76, 2.96])
    smfh_mu = np.array([1.50, 2.90, 2.69, 3.55, 3.03, 2.32, 3.6, 3.7,
                        1.63, 1.50, 2.83, 2.40, 2.05, 3.21, 2.40,
                        3.30, 2.91, 3.41, 3.62, 2.70, 2.38, 2.93, 3.36, 2.61, 2.21, 2.62, 2.81,
                        3.63, 3.64, 3.21, 3.46])

    smfdl_source = np.mean(smfl_mu)
    smfdm_source = np.mean(smfm_mu)
    smfdh_source = np.mean(smfh_mu)
    
    ### post-cap behavior, source from components
    # use beams in flexure
    # assume both class 1 and 2

    # smfd_source = 10.0
    # e-to-d or b-to-a
    seg_smfb_source = np.array([11.0, 6.0])
    seg_smfa_source = np.array([9.0, 4.0])
    steel_b_to_a_ratio = np.mean(np.divide(seg_smfb_source, seg_smfa_source))

    # the bump calibrates to SRG 3rd edition values
    smfe_source = 12.0*0.5
    smfc_source = np.mean(np.array([0.6, 0.2]))
    # mid and high-rises, use HAZUS ratio (not used since source has height typologies)

    ########### steel braced frames
    # (excluding brbfs)
    # sourced from FEMA P-695 studies 
    # ( NIST GCR 10-917-8, Wichman et al. (2021), Chen & Mahin (2010) <- actually sourced into GCR)

    # sbfd_source = np.mean([10.0, 9.0, 9.0, 9.0, 13.0])
    sbfl_mu = np.array([4.3, 6.1, 5.8, 3.0])
    sbfm_mu = np.array([2.4, 3.9])
    sbfh_mu = np.array([3.2, 1.8, 1.4, 1.2])

    sbfdl_source = np.mean(sbfl_mu)
    sbfdm_source = np.mean(sbfm_mu)
    sbfdh_source = np.mean(sbfh_mu)

    ### post-cap behavior, source from components
    # use ratios on buckling braces in tension
    # ratios act on axial tension, but would be similar for drifts
    # for residual capacity, use beams
    # these are typical n values for different shapes at average slenderness

    # the bump calibrates to SRG 3rd edition values
    sbfe_source = 12.0*1.5
    sbfc_source = np.mean(np.array([0.6, 0.2]))
    # mid and high-rises, use HAZUS ratio

    ########### cold formed steel and steel light frames
    # for steel light frame, default to cfs with shear wall
    # this system is not expected to be used often

    # P-695 studies from Pehlivan et al. (2023), Shakeel et al. (2019)
    # should validate with Shamim & Rogers (2015), which is Canada specific (but steel sheathed)
    # 2-3 stories, multiple occupancies, R between 4 and 6.5
    cfs1_mu = np.array([3.5, 3.9, 3.4, 3.9, 3.5, 3.7, 3.5, 3.5,
                        3.6, 3.7, 3.8, 3.6, 3.5, 3.6, 3.2, 3.9, 
                        3.6, 3.7, 3.1, 3.6, 3.1, 3.3, 3.4, 3.2, 
                        3.0, 3.1, 3.1, 3.1,
                        6.9, 6.3, 4.8, 1.9, 7.0, 4.0, 4.6, 
                                 3.9, 2.7, 4.5, 3.3, 4.6, 3.0, 4.1])
    

    # CFS with shear wall
    # use h/b < 2
    # higher rises, apply the ratio on higher h/b ratios
    cfs1d_source = np.mean(cfs1_mu)

    # ratio of b-to-a point
    cfs1a_source = np.array([2.3, 3.4, 2.7, 2.3, 2.9, 5.2, 2.0])
    cfs1b_source = np.array([4.0, 6.5, 4.5, 4.0, 8.2, 13.8, 3.9])
    cfs1_b_to_a_ratio = np.mean(np.divide(cfs1b_source, cfs1a_source))

    cfs1c_source = np.mean([0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.4, 0.2])
    # higher rise
    # d & e: 0.75, c: 1.67


    # CFS with strap braces
    # P-695 study from Fiorino et al. (2017)
    cfs2_mu = np.array([10.7, 9.50, 4.80, 31.2, 11.8, 12.6, 6.70, 
                        6.00, 28.6, 13.5, 6.60, 7.20, 28.6, 6.90])
    cfs2d_source = np.mean(cfs2_mu)

    # ratio of b-to-a point
    cfs2a_source = np.array([6.9, 7.2, 2.3])
    cfs2b_source = np.array([8.4, 10.1, 5.8])
    cfs2_b_to_a_ratio = np.mean(np.divide(cfs2b_source, cfs2a_source))

    cfs2c_source = np.mean([0.8, 0.6, 0.9])

    # collect b-to-a for cfs components

    ########### concrete moment frames
    cmfd_source = 0.025/0.004

    # source: Haselton study in FEMA P-695, Haselton et al. PEER Report 2007/12
    # Haselton et al. PEER report 2007/08, Fazileh et al. (2022) for Canada
    # includes both space frame and perimeter frame archetypes

    # SPECIAL rc moment frames
    # NOT considering ordinary moment frames
    cmfl_mu = np.array([14.0, 19.6, 16.1, 14.0,
                       0.077/0.004, 0.079/0.004, 0.077/0.004, 0.077/0.004, 0.085/0.004, 0.085/0.004, 0.076/0.004, 0.067/0.004, 
                       5.79, 6.27, 5.68, 6.26])
    cmfm_mu = np.array([10.9, 11.3, 13.4, 13.2,
                       0.038/0.004, 0.043/0.004, 0.047/0.004, 0.05/0.004, 0.056/0.004,
                       2.89, 3.99, 4.03, 4.50, 4.16, 3.64])
    cmfh_mu = np.array([9.8, 7.4, 4.1, 7.5, 8.6, 4.4, 3.0, 3.7, 2.8, 3.3,
                       0.023/0.004, 0.028/0.004, 0.035/0.004, 0.024/0.004, 0.031/0.004, 0.019/0.004, 0.021/0.004, 0.026/0.004, 
                        0.022/0.004, 0.024/0.004, 0.020/0.004, 0.022/0.004, 0.016/0.004, 0.018/0.004, 0.023/0.004, 0.018/0.004, 0.023/0.004])

    cmfdl_source = np.mean(cmfl_mu)
    cmfdm_source = np.mean(cmfm_mu)
    cmfdh_source = np.mean(cmfh_mu)
    

    # CMF estimate story drift capacity as beam chord rotation capacity
    # Visnjic (2015) shows this is true up until ~12 stories
    # at 20 stories it drops to about 0.6*beam chord rotation

    # for the actual parameter, can use SEG modelling parameters 
    # for reinforced concrete beam-column joints (plastic rotation angle)
    # take the ratio of point b (end of residual) to point a (start of capacity loss)

    cmfa_source = np.array([0.015, 0.015, 0.015, 0.015, 0.005, 0.005, 0.005, 0.005,
                            0.01, 0.01, 0.01, 0.01, 0.005, 0.005])
    cmfb_source = np.array([0.03, 0.03, 0.025, 0.02, 0.02, 0.015, 0.015, 0.015,
                            0.02, 0.015, 0.02, 0.015, 0.01, 0.01])
    cmf_b_to_a_ratio = np.mean(np.divide(cmfb_source, cmfa_source))
    
    cmfe_source = 0.05/0.004
    cmfc_source = 0.2
    # normalize against a presumed yield of 0.004

    # removing this bump because the sources already account for height archetypes
    # bump is from Visnjic (2015) 
    # low rises get 1.2 bump
    # mid rises 1.0
    # high rises have 0.6 penalty

    ####################### concrete shear walls

    # sourced from NIST GCR 10-917-8 for special concrete shear walls (Gogus & Wallace 2015)
    # Canada source from Nazari & Saatcioglu (2017)

    cswl_mu = np.array([9, 12, 23, 48/5])
    cswm_mu = np.array([22, 19, 25, 17, 85/12, 6/.5])
    cswh_mu = np.array([35, 21, 18, 9, 12, 13, 21, 38, 25, 17, 50, 30, 24, 85/12, 6/.5])

    cswld_source = np.mean(cswl_mu)
    cswmd_source = np.mean(cswm_mu)
    cswhd_source = np.mean(cswh_mu)

    # CSW flexure controlled: take as mid and high rises
    # component: both walls and coupling beams, conforming

    # assume Nud/AgfcE correlates with wvVmcUlt/Avsqrt(fcE)
    # for coupling beam, it's e:d ratio
    cswfd_source = np.array([0.032, 0.026, 0.018, 0.014, 0.032, 0.026, 0.012, 0.011, 0.025, 0.020, 0.03])
    cswfdprime_source = np.array([0.036, 0.030, 0.018, 0.014, 0.036, 0.030, 0.018, 0.014, 0.050, 0.040, 0.050])
    cswfd_ratio = np.mean(np.divide(cswfdprime_source, cswfd_source))
    
    cswfe_source = np.mean([0.04, 0.032, 0.02, 0.014, 0.05, 0.04, 0.05])/0.0025
    cswfc_source = np.mean([0.5, 0.1, 0.0, 0.0, 0.50, 0.50, 0.8])

    # shear controlled walls
    cswsd_source = np.mean([0.01, 0.0075])/0.0025
    cswse_source = np.mean([0.02, 0.01])/0.0025 
    # steep drop off, but not completely vertical
    cswsd_ratio = 1.1
    cswsc_source = np.mean([0.20, 0.0])

    ########### precast
    #  SEG recommends reduction by a factor of 1/2 for walls

    ################## unreinforced masonry
    # sourced from Magenes & Calvi (1997)
    # Kappos et al. (2005)
    # Petry & Beyer (2014)
    urm_mu = np.array([4.25, 2.53, 3.27, 4.63, 1.87, 2.28, 4.54, 7.30, 6.45, 7.55,
                       0.563/0.136, 1.633/0.374, 2.335/0.774, 0.588/0.075, 1.347/0.250, 2.132/0.506, 1.433/0.352])

    # URM: take as the wall, not spandrel, both rocking and sliding
    # assume typical toe crushing drift is 0.5%
    urmd_source = np.mean(urm_mu)
    # urmd_source = np.mean([0.005, 0.004])/0.0025

    # Petry & beyer horizontal loss to peak ratio
    urm_h_to_p_ratio = np.mean(np.array([0.29/0.15, 0.17/0.12, 0.40/0.35, 0.37/0.37, 0.72/0.51, 0.93/0.72, 0.35/0.27,
                                 0.36/0.36, 0.56/0.37, 0.54/0.53, 0.54/0.54, 0.70/0.70]))

    
    urme_source = np.mean([0.005, 0.004])/0.0025
    urmc_source = 0.5 # practically small

    ############################ infill
    # source: Kumar et al (2015), which aggregated a lot of CIW literature
    # mainly focused on CIW as there are no SIW in database
    # also has anchors for yield drift 

    # aggregated sources
    # Fiorato et al 1970, Zarnic and Tomazevic 1984
    # Valiasis et al 1993, Durrani and Haidar 1996
    # Mehrabi et al 1998, Calvi et al 2004
    # Colangelo 2005, Karayannis et al 2005
    # Combescure 2006, Huang et al 2006
    # Kakaletsis and Karayannis 2008, D'Ayala 2009
    
    # table below is the mean from Kumar dissertation
    iw_mu = np.array([3.1, 4.5, 4.3, 4.3, 4.3, 2.5, 1.5, 2.4, 2.6, 
                      2.0, 2.4, 2.2, 2.3, 4.5, 3.6, 3.0, 3.1])
    iwd_source = np.mean(iw_mu)


    # infill RM
    # use infilled frame and masonry infill 
    # includes ductile, both flexible and stiff panels
    # assume aspect ratio of 1.0
    # D point is taken as the onset of residual (1.6*peak)
    # iwd_source = np.mean([1.6*0.0037, 0.01])/0.0025
    # below is mean from kumar dissertation
    iw_e_to_d_ratio  = np.mean(np.array([1.7, 5.0, 4.0, 2.3, 1.9, 3.0, 2.4,
                                 4.3, 2.0, 1.0]))

    # iwe_upc = np.array([np.mean([0.01])/0.0025,
    #                        np.mean([2.3, 4.0])*np.mean([1.0, 2.1]),
    #                        1.0, 4.3*4.0, 4.3*5.0, 2.5*np.mean([1.8, 2.8]), 1.5*1.9,
    #                        np.mean([1.3, 5.0])*np.mean([1.5,5.3]), np.mean([2.0,3.5])*np.mean([1.7*3.0]), 
    #                        1.0, 1.0, 1.0, 1.0, 2.3*4.3, 4.5*2, 1.0, 1.0, 1.0])
    # iwe_source = np.mean(iwe_upc)
    iwc_source = np.mean([0.2, 0.75, 0.4, 0.4, 0.85, 0.45, 0.4, 0.4, 0.8, 0.39, 0.23])

    ############### reinforced masonry

    # from NIST GCR 10-917-8 (Koutromanos & Shing 2010)
    # Mahrous et al (2025) for Canada (normalized against 0.0025 yield drift instead of ~0.005 from that study)
    # Ezzeldin et al. (2016) 
    # group into low and mid/high story 
    # highly dependent on axial load. we'll average between low and high cases
    
    rml_mu = np.array([5.2, 8.1, 5.2, 8.3, 13.3, 14.4, 13.3, 14.4,
                       5.20, 13.25, 8.10, 16.0, 13.30, 19.0, 14.40, 24.36])
    rmh_mu = np.array([11.8, 6.4, 14.6, 28.4, 7.1, 16.0, 11.3, 13.6, 42.8, 29.0, 17.7, 20.7,
                       0.027/0.0025, 0.012/0.0025, 0.0287/0.0025, 0.0119/0.0025, 
                       11.80, 14.50, 6.40, 10.67, 14.60, 22.50, 28.40, 34.80, 7.10, 11.80, 16.0, 24.32])
    rmld_source = np.mean(rml_mu)
    rmhd_source = np.mean(rmh_mu)

    rmd_source = 0.01/0.0025


    # post-cap parameters
    # use shear-governed critical drift ratios
    # take the ratio of r to m point (beginning of residual to beginning of strength loss)
    # c to m ratio is used to hard code the end of the residual regime in nmfs.py
    rme_source = 0.04/0.0025
    r_to_m_ratio = 0.01 / 0.005
    rmc_source = 0.5

    # this represents the highest ductility available
    # any reductions would be applied by the ductility level, tied to code era
    ductility_df = pd.DataFrame({
        "WLF-P9":   [wlfp9_d_source, wood_e_to_d_ratio * wlfp9_d_source, wc_source],
        "WLF":      [wd_source,     wood_e_to_d_ratio * wd_source,      wc_source],
        "WPB":      [wpb_d_source,  wood_e_to_d_ratio * wpb_d_source,   wc_source],
        'SMFL' : [smfdl_source, steel_b_to_a_ratio*smfdl_source, smfc_source],
        'SMFM' : [smfdm_source, steel_b_to_a_ratio*smfdm_source, smfc_source],
        'SMFH' : [smfdh_source, steel_b_to_a_ratio*smfdh_source, smfc_source],
        'SBFL' : [sbfdl_source, sbfe_source, sbfc_source],
        'SBFM' : [sbfdm_source, sbfe_source, sbfc_source],
        'SBFH' : [sbfdh_source, sbfe_source, sbfc_source],
        'SLF' : [cfs2d_source, cfs2_b_to_a_ratio*cfs2d_source, cfs2c_source],
        'SCWL' : [cswld_source, cswsd_ratio*cswld_source, cswfc_source], # use flexure ctrl shear wall
        'SCWM' : [cswmd_source, cswfd_ratio*cswmd_source, cswfc_source], # use flexure ctrl shear wall
        'SCWH' : [cswhd_source, cswfd_ratio*cswhd_source, cswfc_source], # use flexure ctrl shear wall
        'SIWL' : [iwd_source, iw_e_to_d_ratio*iwd_source, iwc_source],
        'SIWM' : [0.66*iwd_source, iw_e_to_d_ratio*0.66*iwd_source, iwc_source],
        'SIWH' : [0.5*iwd_source, iw_e_to_d_ratio*0.5*iwd_source, iwc_source],
        'CMFL' : [cmfdl_source, cmf_b_to_a_ratio*cmfdl_source, cmfc_source],
        'CMFM' : [cmfdm_source, cmf_b_to_a_ratio*cmfdm_source, cmfc_source],
        'CMFH' : [cmfdh_source, cmf_b_to_a_ratio*cmfdh_source, cmfc_source],
        'CSWL' : [cswld_source, cswsd_ratio*cswld_source, cswsc_source], # shear control
        'CSWM' : [cswmd_source, cswfd_ratio*cswmd_source, cswfc_source], # flexure control
        'CSWH' : [cswhd_source, cswfd_ratio*cswhd_source, cswfc_source], # flexure control
        'CIWL' : [iwd_source, iw_e_to_d_ratio*iwd_source, iwc_source],
        'CIWM' : [0.66*iwd_source, iw_e_to_d_ratio*0.66*iwd_source, iwc_source],
        'CIWH' : [0.5*iwd_source, iw_e_to_d_ratio*0.5*iwd_source, iwc_source],
        'PCWL' : [0.5*cswld_source, 0.5*cswsd_ratio*cswld_source, cswsc_source], # shear control, assume CSW
        'PCWM' : [0.5*cswmd_source, 0.5*cswfd_ratio*cswmd_source, cswfc_source], # flexure control, assume CSW
        'PCWH' : [0.5*cswhd_source, 0.5*cswfd_ratio*cswhd_source, cswfc_source], # flexure control, assume CSW
        'PCF1L' : [0.5*cswld_source, 0.5*cswsd_ratio*cswld_source, cswsc_source], # shear control, assume CSW
        'PCF1M' : [0.5*cswmd_source, 0.5*cswfd_ratio*cswmd_source, cswfc_source], # flexure control, assume CSW
        'PCF1H' : [0.5*cswhd_source, 0.5*cswfd_ratio*cswhd_source, cswfc_source], # flexure control, assume CSW
        'PCF2L' : [cmfdl_source, cmf_b_to_a_ratio*cmfdl_source, cmfc_source], # assume CMF
        'PCF2M' : [cmfdm_source, cmf_b_to_a_ratio*cmfdm_source, cmfc_source],     # assume CMF
        'PCF2H' : [cmfdh_source, cmf_b_to_a_ratio*cmfdh_source, cmfc_source], # assume CMF
        'RMLL'  : [rmld_source, r_to_m_ratio*rmld_source, rmc_source],
        'RMLM'  : [rmhd_source, r_to_m_ratio*rmhd_source, rmc_source],
        'RMLH'  : [rmhd_source, r_to_m_ratio*rmhd_source, rmc_source],
        'RMCL'  : [rmld_source, r_to_m_ratio*rmld_source, rmc_source],
        'RMCM'  : [rmhd_source, r_to_m_ratio*rmhd_source, rmc_source],
        'RMCH'  : [rmhd_source, r_to_m_ratio*rmhd_source, rmc_source],
        'URML'  : [urmd_source, urm_h_to_p_ratio*urmd_source, urmc_source],
        'URMM'  : [0.66*urmd_source, urm_h_to_p_ratio*urmd_source, urmc_source],
        'URMH'  : [0.66*urmd_source, urm_h_to_p_ratio*urmd_source, urmc_source],
        'CFS1L' : [cfs1d_source, cfs1_b_to_a_ratio*cfs1d_source, cfs1c_source],
        'CFS1M' : [0.75*cfs1d_source, cfs1_b_to_a_ratio*0.75*cfs1d_source, cfs1c_source],
        'CFS1H' : [0.75*cfs1d_source, cfs1_b_to_a_ratio*0.75*cfs1d_source, cfs1c_source],
        'CFS2' : [cfs2d_source, cfs2_b_to_a_ratio*cfs2d_source, cfs2c_source],
    },
    index=['d_peak', 'e_plateau', 'c_residual'], dtype=float)

    # ensure post-cap is at least ultimate ductility
    ductility_df.loc["e_plateau"] = ductility_df.loc["e_plateau"].clip(lower=ductility_df.loc["d_peak"])

    return ductility_df

SEG_BASE_POST_PEAK_PARAMS = get_p695_ductility()


def get_p695_overstrength():
    '''
    FEMA P-695 Omega_0 factors for each SFRS, compiled from literature review
    of appropriate analytical studies.

    TODO:
    Note that FEMA recommends that this limited to 3.0 and need not exceed 1.5 R
    '''
    # design-to-peak overstrength
    
    wlf_omega = np.array([2.0, 2.5, 2.0, 3.7, 2.9, 2.6, 2.1, 3.6, 4.0, 4.0, 3.4, 3.3,
                          3.64, 3.13, 2.0, 2.90, 2.41, 1.89,
                          3.32, 3.42, 3.53, 3.64, 2.84, 3.13, 3.22, 3.36,
                        7.78, 8.02, 8.23, 8.47, 4.06, 4.41, 4.51, 4.71,
                          3.13, 1.53, 2.00, 1.29, 4.41, 1.55])
    wlfp9_omega = np.array([4.1, 3.8, 5.4, 3.5, 
                           12.10, 8.31, 5.00, 4.93, 8.72, 5.54, 3.76,
                                     11.17, 11.51, 11.82, 12.10, 4.57, 4.73, 4.88, 5.00, 5.13, 5.36,
                           12.10, 5.18, 8.73, 8.31, 2.73, 3.02, 5.00, 2.22, 3.82, 4.93, 2.79, 3.21])
    wpb_omega = 1.5*np.array([54.5/52.8, 76.9/70.8, 32.4/28.6, 27.4/25.7])


    smfl_omega = np.array([4.63, 3.69, 2.88, 5.46, 3.94, 2.21, 2.78, 2.22,
                           4.64, 3.68, 2.55, 3.45, 5.48, 3.93, 2.84, 3.58])
    smfm_omega = np.array([4.45, 2.55, 3.44, 4.71, 3.27, 2.68, 2.27, 2.84, 2.61, 1.85,
                           2.87, 4.71, 2.19, 3.25,
                           2.54, 2.86, 2.00, 2.31, 1.27, 1.46, 1.24, 1.43,
                           2.00, 2.29, 2.41, 2.57])
    smfh_omega = np.array([4.67, 3.59, 3.27, 2.71, 2.48, 2.61, 1.92, 1.68,
                           4.53, 4.67, 3.26, 2.65, 2.23, 2.69, 2.44, 2.57,
                           1.91, 2.22, 1.33, 1.50,
                           2.63, 3.31, 3.32, 3.52, 2.09, 2.52, 2.52, 2.77, 1.89, 2.27, 2.34, 2.39])

    sbfl_omega = np.array([1.44, 1.41, 1.38, 2.41])
    sbfm_omega = np.array([1.28, 1.86])
    sbfh_omega = np.array([1.60, 2.11, 2.18, 1.56])

    
    cfs1_omega = np.array([2.3, 3.1, 3.8, 3.0, 3.3, 3.7, 2.2, 3.2,
                           3.4, 3.1, 3.0, 3.5, 2.1, 3.2, 3.2, 3.0,
                           3.1, 3.2, 2.2, 3.4, 3.7, 2.1, 3.1 ,3.8, 
                           3.2, 3.8, 3.0, 3.3,
                           1.06, 1.17, 1.27, 1.16, 1.33, 1.23, 1.32, 
                                    1.17, 1.16, 1.31, 1.21, 1.10, 1.17, 1.34])

    cfs2_omega = np.array([1.42, 1.54, 1.55, 1.80, 1.54, 1.33, 1.49, 1.67, 2.13, 1.37, 1.46, 1.52, 1.62])


    cmfl_omega = np.array([1.6, 1.8, 1.7, 4.0, 3.5, 3.1,
                           4.0, 4.9, 4.0, 1.6, 3.5, 4.4, 3.1, 1.8,
                           1.60, 1.43, 2.86, 1.34, 1.59])
    cmfm_omega = np.array([1.6, 2.7, 1.6, 3.3,
                          1.6, 1.7, 2.7, 1.6, 3.3,
                          1.44, 1.55, 1.31, 1.34, 2.29, 2.01])
    cmfh_omega = np.array([1.6, 1.7, 2.6, 2.3, 2.8, 3.5, 1.8, 1.8, 1.8, 3.4,
                          1.6, 2.3, 2.6, 3.3, 2.9, 2.9, 2.7,
                            1.7, 2.1, 2.1, 3.2, 2.5, 2.8, 2.5, 2.4, 1.6, 2.0])

    cswl_omega = np.array([2.39, 1.82, 1.72, 1.2])
    cswm_omega = np.array([1.45, 1.63, 2.98, 1.83, 1.1, 50/45])
    cswh_omega = np.array([2.91, 4.44, 4.30, 2.39, 1.82, 2.03, 1.98, 1.50, 2.98, 1.83, 1.85, 5.64, 3.08, 1.1, 50/45])


    
    
    rml_omega = np.array([1.84, 2.28, 1.62, 2.61, 1.84, 1.82, 1.62, 1.80,
                          1.63, 1.78, 2.34, 1.99, 2.27, 2.67, 2.95,
                          1.63, 1.48, 1.90, 2.40, 1.99, 2.27, 2.10, 2.56])
    rmh_omega = np.array([1.87, 1.89, 1.61, 1.65, 1.93, 1.68, 1.73, 1.59, 1.47, 1.41, 1.64, 1.46,
                          3.6, 3.3, 3.4, 2.6,
                          1.67, 1.80, 1.54, 1.68, 1.49, 1.59, 1.48, 1.59, 1.55, 1.66, 1.32, 1.42,
                          1.47, 1.76, 1.15, 1.34, 1.15, 1.24, 1.47, 1.76, 1.64, 1.42, 1.16, 1.20])

    # capacity curve info from Kappos et al. (2005)
    urm_omega = np.array([0.328/0.320, 0.214/0.189, 0.158/0.135, 0.248/0.231, 0.164/0.135, 0.111/0.092, 0.204/0.184])

    # Kumar et al. calculated these from yield. This is handled in estimate_SEG_peak_to_yield_overstrength
    iw_omega = np.array([1/0.8, 1/0.8, 1/0.6, 1/0.8, 1/0.8, 1/0.9, 1/0.85, 1/0.95, 1/0.9, 1/0.9, 1/0.8, 1/0.9, 1/0.6, 1/0.8, 1/0.75])


    omega_dict =  {
        "WLF-P9":   np.mean(wlfp9_omega),
        "WLF":      np.mean(wlf_omega),
        "WPB":      np.mean(wpb_omega),
        "SMFL":     np.mean(smfl_omega),
        "SMFM":     np.mean(smfm_omega),
        "SMFH":     np.mean(smfh_omega),
        "SBFL":     np.mean(sbfl_omega),
        "SBFM":     np.mean(sbfm_omega),
        "SBFH":     np.mean(sbfh_omega),
        "SLF":      np.mean(cfs2_omega),
        "SCWL":     np.mean(cswl_omega),
        "SCWM":     np.mean(cswm_omega),
        "SCWH":     np.mean(cswh_omega),
        "SIWL":     np.mean(iw_omega),
        "SIWM":     np.mean(iw_omega),
        "SIWH":     np.mean(iw_omega),
        "CMFL":     np.mean(cmfl_omega),
        "CMFM":     np.mean(cmfm_omega),
        "CMFH":     np.mean(cmfh_omega),
        "CSWL":     np.mean(cswl_omega),
        "CSWM":     np.mean(cswm_omega),
        "CSWH":     np.mean(cswh_omega),
        "CIWL":     np.mean(iw_omega),
        "CIWM":     np.mean(iw_omega),
        "CIWH":     np.mean(iw_omega),
        "PCWL":     np.mean(cswl_omega),
        "PCWM":     np.mean(cswm_omega),
        "PCWH":     np.mean(cswh_omega),
        "PCF1L":    np.mean(cswl_omega),
        "PCF1M":    np.mean(cswm_omega),
        "PCF1H":    np.mean(cswh_omega),
        "PCF2L":    np.mean(cmfl_omega),
        "PCF2M":    np.mean(cmfm_omega),
        "PCF2H":    np.mean(cmfh_omega),
        "RMLL":     np.mean(rml_omega),
        "RMLM":     np.mean(rmh_omega),
        "RMLH":     np.mean(rmh_omega),
        "RMCL":     np.mean(rml_omega),
        "RMCM":     np.mean(rmh_omega),
        "RMCH":     np.mean(rmh_omega),
        "URML":     np.mean(urm_omega),
        "URMM":     np.mean(urm_omega),
        "URMH":     np.mean(urm_omega),
        "CFS1L":    np.mean(cfs1_omega),
        "CFS1M":    np.mean(cfs1_omega),
        "CFS1H":    np.mean(cfs1_omega),
        "CFS2":     np.mean(cfs2_omega),
        }

    return {k: np.minimum(v, 3.0) for k, v in omega_dict.items()}

P695_BASE_OVERSTRENGTH_PARAMS = get_p695_overstrength()


def estimate_SEG_peak_to_yield_overstrength(lfrs, number_of_stories, Ro):
    '''
    Use FEMA P-695 (peak-to-design) Omega and SEG's (yield-to-design) overstrength
    to back-calculate the yield-to-peak overstrength. This is meant to replace
    Hazus Omega_p value.
    '''
    height_conditions = [
        (number_of_stories < 4),
        (number_of_stories < 7) & (number_of_stories >= 4),
        (number_of_stories > 7),
    ]

    height_archetypes = ['L', 'M', 'H']
    no_height_delin = ["WLF-P9", "WLF", "WPB", "SLF", "CFS2"]

    height_suffix = np.select(height_conditions, height_archetypes, default="")
    if lfrs not in no_height_delin:
        lfrsh = lfrs + height_suffix
    else:
        lfrsh = lfrs 

    p695_peak_to_design_Omega = P695_BASE_OVERSTRENGTH_PARAMS[lfrsh]

    if lfrs in ['CIW', 'SIW']:
        return np.maximum(p695_peak_to_design_Omega, 1.0)
    else:
        return np.maximum(p695_peak_to_design_Omega/Ro, 1.0)

def estimate_SEG_ductility(lfrs, number_of_stories, code_level, Rd_2025, Rd_historical):
    '''
    Use a best-guess estimate of ductility (see SEG_BASE_POST_PEAK_PARAMS), 
    adjusted by SEG's correction for past code years to characterize the 
    ratio of ultimate to yield drifts.

    '''
    height_conditions = [
        (number_of_stories < 4),
        (number_of_stories < 7) & (number_of_stories >= 4),
        (number_of_stories > 7),
    ]

    height_archetypes = ['L', 'M', 'H']
    no_height_delin = ["WLF-P9", "WLF", "WPB", "SLF", "CFS2"]

    height_suffix = np.select(height_conditions, height_archetypes, default="")
    if lfrs not in no_height_delin:
        lfrsh = lfrs + height_suffix
    else:
        lfrsh = lfrs 

    base_ductility = SEG_BASE_POST_PEAK_PARAMS.loc['d_peak', lfrsh]
    # period_adjustment_factor = (Rd_historical - 0.9) / (Rd_2025 - 0.9)
    
    # another idea for period adjustment
    # 1) get era of building
    # 2) get code-era ductility classification (ductile, moderate, limited, conventional)
    # 3) use the ratio of <code_era_ductility>/<high-code_ductility> Rd as the period adjustment factor
    ductility_level = DUCTILITY_LOOKUP_TABLE.loc[code_level, lfrs]
    high_code_ductility = DUCTILITY_LOOKUP_TABLE.loc['high-code', lfrs]
    code_era_Rd = R_D_TABLE(lfrs, ductility_level=ductility_level)
    high_code_Rd = R_D_TABLE(lfrs, ductility_level=high_code_ductility)
    period_adjustment_factor = code_era_Rd/high_code_Rd

    # period_adjustment_factor = 1.0

    return np.maximum(base_ductility * period_adjustment_factor, 1.1)


def estimate_SEG_postcap(lfrs, number_of_stories, code_level, Rd_2025, Rd_historical):
    '''
    Use a best-guess estimate of post-cap (see SEG_BASE_POST_PEAK_PARAMS), 
    adjusted by SEG's correction for past code years to characterize the 
    ratio of residual-regime to yield drifts.

    '''
    height_conditions = [
        (number_of_stories < 4),
        (number_of_stories < 7) & (number_of_stories >= 4),
        (number_of_stories > 7),
    ]

    height_archetypes = ['L', 'M', 'H']
    no_height_delin = ["WLF-P9", "WLF", "WPB", "SLF", "CFS2"]

    height_suffix = np.select(height_conditions, height_archetypes, default="")
    if lfrs not in no_height_delin:
        lfrsh = lfrs + height_suffix
    else:
        lfrsh = lfrs 

    base_postcap = SEG_BASE_POST_PEAK_PARAMS.loc['e_plateau', lfrsh]
    # period_adjustment_factor = (Rd_historical - 0.9) / (Rd_2025 - 0.9)

    # ductility_level = DUCTILITY_LOOKUP_TABLE.loc[code_level, lfrs]
    # high_code_ductility = DUCTILITY_LOOKUP_TABLE.loc['high-code', lfrs]
    # code_era_Rd = R_D_TABLE(lfrs, ductility_level=ductility_level)
    # high_code_Rd = R_D_TABLE(lfrs, ductility_level=high_code_ductility)
    # period_adjustment_factor = code_era_Rd/high_code_Rd

    # if tied to ductility, no need for additional penalty
    period_adjustment_factor = 1.0

    return np.maximum(base_postcap * period_adjustment_factor, 1.1)

def estimate_SEG_residual(lfrs, number_of_stories):
    '''
    Use a best-guess estimate of residual (see SEG_BASE_POST_PEAK_PARAMS).

    '''
    height_conditions = [
        (number_of_stories < 4),
        (number_of_stories < 7) & (number_of_stories >= 4),
        (number_of_stories > 7),
    ]

    height_archetypes = ['L', 'M', 'H']
    no_height_delin = ["WLF-P9", "WLF", "WPB", "SLF", "CFS2"]

    height_suffix = np.select(height_conditions, height_archetypes, default="")
    if lfrs not in no_height_delin:
        lfrsh = lfrs + height_suffix
    else:
        lfrsh = lfrs 

    residual_ratio = SEG_BASE_POST_PEAK_PARAMS.loc['c_residual', lfrsh]

    return residual_ratio

# index is HAZUS typology
# returns C_s, design strength as fraction of building weight
HAZUS_DESIGN_STRENGTH_TABLE = pd.DataFrame({
    "W1": [0.200,  0.150,  0.100,  0.100 ],
    "W2": [0.200,  0.100,  0.050,  0.050 ],
    "S1L": [0.133,  0.067,  0.033,  0.033 ],
    "S1M": [0.100,  0.050,  0.025,  0.025 ],
    "S1H": [0.067,  0.033,  0.017,  0.017 ],
    "S2L": [0.200,  0.100,  0.050,  0.050 ],
    "S2M": [0.200,  0.100,  0.050,  0.050 ],
    "S2H": [0.150,  0.075,  0.038,  0.038 ],
    "S3": [0.200,  0.100,  0.050,  0.050 ],
    "S4L": [0.160,  0.080,  0.040,  0.040 ],
    "S4M": [0.160,  0.080,  0.040,  0.040 ],
    "S4H": [0.120,  0.060,  0.030,  0.030 ],
    "S5L": [0.050,  0.050,  0.050,  0.050 ],
    "S5M": [0.050,  0.050,  0.050,  0.050 ],
    "S5H": [0.038,  0.038,  0.038,  0.038 ],
    "C1L": [0.133,  0.067,  0.033,  0.033 ],
    "C1M": [0.133,  0.067,  0.033,  0.033 ],
    "C1H": [0.067,  0.033,  0.017,  0.017 ],
    "C2L": [0.200,  0.100,  0.050,  0.050 ],
    "C2M": [0.200,  0.100,  0.050,  0.050 ],
    "C2H": [0.150,  0.075,  0.038,  0.038 ],
    "C3L": [0.050,  0.050,  0.050,  0.050 ],
    "C3M": [0.050,  0.050,  0.050,  0.050 ],
    "C3H": [0.038,  0.038,  0.038,  0.038 ],
    "PC1": [0.200,  0.100,  0.050,  0.050 ],
    "PC2L": [0.200,  0.100,  0.050,  0.050 ],
    "PC2M": [0.200,  0.100,  0.050,  0.050 ],
    "PC2H": [0.150,  0.075,  0.038,  0.038 ],
    "RM1L": [0.267,  0.133,  0.067,  0.067 ],
    "RM1M": [0.267,  0.133,  0.067,  0.067 ],
    "RM2L": [0.267,  0.133,  0.067,  0.067 ],
    "RM2M": [0.267,  0.133,  0.067,  0.067 ],
    "RM2H": [0.200,  0.100,  0.050,  0.050 ],
    "URML": [0.067,  0.067,  0.067,  0.067 ],
    "URMM": [0.067,  0.067,  0.067,  0.067 ],
    "URMH": [0.067,  0.067,  0.067,  0.067 ],
}, index=['high-code', 'moderate-code', 'low-code', 'pre-code'], dtype=float)

# returns modal factors
# alpha_1: weight, fraction of weight effective in pushover
# alpha_2: height, fraction of height at location of pushover mode displacement
# overstrength_ratios
# gamma: yield, ratio of yield to design strength
# lambda: ultimate, ratio of ultimate to yield strength
HAZUS_PUSHOVER_TABLE = pd.DataFrame({
    "W1": [0.75, 0.75, 1.50, 3.00],
    "W2": [0.75, 0.75, 1.50, 2.50],
    "S1L": [0.80, 0.75, 1.50, 3.00],
    "S1M": [0.80, 0.75, 1.25, 3.00],
    "S1H": [0.75, 0.60, 1.10, 3.00],
    "S2L": [0.75, 0.75, 1.50, 2.00],
    "S2M": [0.75, 0.75, 1.25, 2.00],
    "S2H": [0.65, 0.60, 1.10, 2.00],
    "S3": [0.75, 0.75, 1.50, 2.00],
    "S4L": [0.75, 0.75, 1.50, 2.25],
    "S4M": [0.75, 0.75, 1.25, 2.25],
    "S4H": [0.65, 0.60, 1.10, 2.25],
    "S5L": [0.75, 0.75, 1.50, 2.00],
    "S5M": [0.75, 0.75, 1.25, 2.00],
    "S5H": [0.65, 0.60, 1.10, 2.00],
    "C1L": [0.80, 0.75, 1.50, 3.00],
    "C1M": [0.80, 0.75, 1.25, 3.00],
    "C1H": [0.75, 0.60, 1.10, 3.00],
    "C2L": [0.75, 0.75, 1.50, 2.50],
    "C2M": [0.75, 0.75, 1.25, 2.50],
    "C2H": [0.65, 0.60, 1.10, 2.50],
    "C3L": [0.75, 0.75, 1.50, 2.25],
    "C3M": [0.75, 0.75, 1.25, 2.25],
    "C3H": [0.65, 0.60, 1.10, 2.25],
    "PC1": [0.50, 0.75, 1.50, 2.00],
    "PC2L": [0.75, 0.75, 1.50, 2.00],
    "PC2M": [0.75, 0.75, 1.25, 2.00],
    "PC2H": [0.65, 0.60, 1.10, 2.00],
    "RM1L": [0.75, 0.75, 1.50, 2.00],
    "RM1M": [0.75, 0.75, 1.25, 2.00],
    "RM2L": [0.75, 0.75, 1.50, 2.00],
    "RM2M": [0.75, 0.75, 1.25, 2.00],
    "RM2H": [0.65, 0.60, 1.10, 2.00],
    "URML": [0.50, 0.75, 1.50, 2.00],
    "URMM": [0.75, 0.75, 1.25, 2.00],
    "URMH": [0.75, 0.75, 1.25, 2.00],
}, 
index=['alpha_1', 'alpha_2', 'gamma', 'lambda'], dtype=float)

# returns ductility factor mu
# ratio of ultimate displacement to lambda * yield displacement
HAZUS_DUCTILITY_TABLE = pd.DataFrame({
    "W1":   [8.0, 6.0, 6.0, 6.0],
    "W2":   [8.0, 6.0, 6.0, 6.0],
    "S1L":  [8.0, 6.0, 5.0, 5.0],
    "S1M":  [5.3, 4.0, 3.3, 3.3],
    "S1H":  [4.0, 3.0, 2.5, 2.5],
    "S2L":  [8.0, 6.0, 5.0, 5.0],
    "S2M":  [5.3, 4.0, 3.3, 3.3],
    "S2H":  [4.0, 3.0, 2.5, 2.5],
    "S3":   [8.0, 6.0, 5.0, 5.0],
    "S4L":  [8.0, 6.0, 5.0, 5.0],
    "S4M":  [5.3, 4.0, 3.3, 3.3],
    "S4H":  [4.0, 3.0, 2.5, 2.5],
    "S5L":  [5.0, 5.0, 5.0, 5.0],
    "S5M":  [3.3, 3.3, 3.3, 3.3],
    "S5H":  [2.5, 2.5, 2.5, 2.5],
    "C1L":  [8.0, 6.0, 5.0, 5.0],
    "C1M":  [5.3, 4.0, 3.3, 3.3],
    "C1H":  [4.0, 3.0, 2.5, 2.5],
    "C2L":  [8.0, 6.0, 5.0, 5.0],
    "C2M":  [5.3, 4.0, 3.3, 3.3],
    "C2H":  [4.0, 3.0, 2.5, 2.5],
    "C3L":  [5.0, 5.0, 5.0, 5.0],
    "C3M":  [3.3, 3.3, 3.3, 3.3],
    "C3H":  [2.5, 2.5, 2.5, 2.5],
    "PC1":  [8.0, 6.0, 5.0, 5.0],
    "PC2L": [8.0, 6.0, 5.0, 5.0],
    "PC2M": [5.3, 4.0, 3.3, 3.3],
    "PC2H": [4.0, 3.0, 2.5, 2.5],
    "RM1L": [8.0, 6.0, 5.0, 5.0],
    "RM1M": [5.3, 4.0, 3.3, 3.3],
    "RM2L": [8.0, 6.0, 5.0, 5.0],
    "RM2M": [5.3, 4.0, 3.3, 3.3],
    "RM2H": [4.0, 3.0, 2.5, 2.5],
    "URML": [5.0, 5.0, 5.0, 5.0],
    "URMM": [3.3, 3.3, 3.3, 3.3],
    "URMH": [3.3, 3.3, 3.3, 3.3],
}, index=['high-code', 'moderate-code', 'low-code', 'pre-code'], dtype=float)

# for each typology and code-level
# D are spectral displacement values
# A are spectral acceleration values
# return yield capacities
# D_y in meters
# A_y in g
# ultimate capacities
# D_u in meters
# A_y in g

high_code_cc_table = pd.DataFrame({
    "W1":   [0.48*units.inch, 0.400, 11.51*units.inch, 1.200],
    "W2":   [0.63*units.inch, 0.400, 12.53*units.inch, 1.000],
    "S1L":  [0.61*units.inch, 0.250, 14.67*units.inch, 0.749],
    "S1M":  [1.78*units.inch, 0.156, 28.40*units.inch, 0.468],
    "S1H":  [4.66*units.inch, 0.098, 55.88*units.inch, 0.293],
    "S2L":  [0.63*units.inch, 0.400, 10.02*units.inch, 0.800],
    "S2M":  [2.43*units.inch, 0.333, 25.88*units.inch, 0.667],
    "S2H":  [7.75*units.inch, 0.254, 61.97*units.inch, 0.508],
    "S3":   [0.63*units.inch, 0.400, 10.02*units.inch, 0.800],
    "S4L":  [0.38*units.inch, 0.320, 6.91 *units.inch, 0.720],
    "S4M":  [1.09*units.inch, 0.267, 13.10*units.inch, 0.600],
    "S4H":  [3.49*units.inch, 0.203, 31.37*units.inch, 0.457],
    "S5L":  [0.12*units.inch, 0.100, 1.20 *units.inch, 0.200],
    "S5M":  [0.34*units.inch, 0.083, 2.27 *units.inch, 0.167],
    "S5H":  [1.09*units.inch, 0.063, 5.45 *units.inch, 0.127],
    "C1L":  [0.39*units.inch, 0.250, 9.39 *units.inch, 0.749],
    "C1M":  [1.15*units.inch, 0.208, 18.44*units.inch, 0.624],
    "C1H":  [2.01*units.inch, 0.098, 24.13*units.inch, 0.293],
    "C2L":  [0.48*units.inch, 0.400, 9.59 *units.inch, 1.000],
    "C2M":  [1.04*units.inch, 0.333, 13.84*units.inch, 0.833],
    "C2H":  [2.94*units.inch, 0.254, 29.39*units.inch, 0.635],
    "C3L":  [0.12*units.inch, 0.100, 1.35 *units.inch, 0.225],
    "C3M":  [0.26*units.inch, 0.083, 1.95 *units.inch, 0.188],
    "C3H":  [0.74*units.inch, 0.063, 4.13 *units.inch, 0.143],
    "PC1":  [0.72*units.inch, 0.600, 11.51*units.inch, 1.200],
    "PC2L": [0.48*units.inch, 0.400, 7.67 *units.inch, 0.800],
    "PC2M": [1.04*units.inch, 0.333, 11.07*units.inch, 0.667],
    "PC2H": [2.94*units.inch, 0.254, 23.52*units.inch, 0.508],
    "RM1L": [0.64*units.inch, 0.533, 10.23*units.inch, 1.066],
    "RM1M": [1.38*units.inch, 0.444, 14.76*units.inch, 0.889],
    "RM2L": [0.64*units.inch, 0.533, 10.23*units.inch, 1.066],
    "RM2M": [1.38*units.inch, 0.444, 14.76*units.inch, 0.889],
    "RM2H": [3.92*units.inch, 0.338, 31.35*units.inch, 0.677],
    "URML": [0.24*units.inch, 0.200, 2.40 *units.inch, 0.400],
    "URMM": [0.27*units.inch, 0.111, 1.81 *units.inch, 0.222],
    "URMH": [0.27*units.inch, 0.111, 1.81 *units.inch, 0.222],
}, index=["D_y", "A_y", "D_u", "A_u"], dtype=float)

moderate_code_cc_table = pd.DataFrame({
    "W1":   [0.36*units.inch, 0.300, 6.48 *units.inch, 0.900],
    "W2":   [0.31*units.inch, 0.200, 4.70 *units.inch, 0.500],
    "S1L":  [0.31*units.inch, 0.125, 5.50 *units.inch, 0.375],
    "S1M":  [0.89*units.inch, 0.078, 10.65*units.inch, 0.234],
    "S1H":  [2.33*units.inch, 0.049, 20.96*units.inch, 0.147],
    "S2L":  [0.31*units.inch, 0.200, 3.76 *units.inch, 0.400],
    "S2M":  [1.21*units.inch, 0.167, 9.70 *units.inch, 0.333],
    "S2H":  [3.87*units.inch, 0.127, 23.24*units.inch, 0.254],
    "S3":   [0.31*units.inch, 0.200, 3.76 *units.inch, 0.400],
    "S4L":  [0.19*units.inch, 0.160, 2.59 *units.inch, 0.360],
    "S4M":  [0.55*units.inch, 0.133, 4.91 *units.inch, 0.300],
    "S4H":  [1.74*units.inch, 0.102, 11.76*units.inch, 0.228],
    "S5L":  [0.12*units.inch, 0.100, 1.20 *units.inch, 0.200],
    "S5M":  [0.34*units.inch, 0.083, 2.27 *units.inch, 0.167],
    "S5H":  [1.09*units.inch, 0.063, 5.45 *units.inch, 0.127],
    "C1L":  [0.20*units.inch, 0.125, 3.52 *units.inch, 0.375],
    "C1M":  [0.58*units.inch, 0.104, 6.91 *units.inch, 0.312],
    "C1H":  [1.01*units.inch, 0.049, 9.05 *units.inch, 0.147],
    "C2L":  [0.24*units.inch, 0.200, 3.60 *units.inch, 0.500],
    "C2M":  [0.52*units.inch, 0.167, 5.19 *units.inch, 0.417],
    "C2H":  [1.47*units.inch, 0.127, 11.02*units.inch, 0.317],
    "C3L":  [0.12*units.inch, 0.100, 1.35 *units.inch, 0.225],
    "C3M":  [0.26*units.inch, 0.083, 1.95 *units.inch, 0.188],
    "C3H":  [0.74*units.inch, 0.063, 4.13 *units.inch, 0.143],
    "PC1":  [0.36*units.inch, 0.300, 4.32 *units.inch, 0.600],
    "PC2L": [0.24*units.inch, 0.200, 2.88 *units.inch, 0.400],
    "PC2M": [0.52*units.inch, 0.167, 4.15 *units.inch, 0.333],
    "PC2H": [1.47*units.inch, 0.127, 8.82 *units.inch, 0.254],
    "RM1L": [0.32*units.inch, 0.267, 3.84 *units.inch, 0.533],
    "RM1M": [0.69*units.inch, 0.222, 5.54 *units.inch, 0.444],
    "RM2L": [0.32*units.inch, 0.267, 3.84 *units.inch, 0.533],
    "RM2M": [0.69*units.inch, 0.222, 5.54 *units.inch, 0.444],
    "RM2H": [1.96*units.inch, 0.169, 11.76*units.inch, 0.338],
    "URML": [0.24*units.inch, 0.200, 2.40 *units.inch, 0.400],
    "URMM": [0.27*units.inch, 0.111, 1.81 *units.inch, 0.222],
    "URMH": [0.27*units.inch, 0.111, 1.81 *units.inch, 0.222],
}, index=["D_y", "A_y", "D_u", "A_u"], dtype=float)

low_code_cc_table = pd.DataFrame({
    "W1":   [0.24*units.inch, 0.200, 4.32*units.inch, 0.600],
    "W2":   [0.16*units.inch, 0.100, 2.35*units.inch, 0.250],
    "S1L":  [0.15*units.inch, 0.062, 2.29*units.inch, 0.187],
    "S1M":  [0.44*units.inch, 0.039, 4.44*units.inch, 0.117],
    "S1H":  [1.16*units.inch, 0.024, 8.73*units.inch, 0.073],
    "S2L":  [0.16*units.inch, 0.100, 1.57*units.inch, 0.200],
    "S2M":  [0.61*units.inch, 0.083, 4.04*units.inch, 0.167],
    "S2H":  [1.94*units.inch, 0.063, 9.68*units.inch, 0.127],
    "S3":   [0.16*units.inch, 0.100, 1.57*units.inch, 0.200],
    "S4L":  [0.10*units.inch, 0.080, 1.08*units.inch, 0.180],
    "S4M":  [0.27*units.inch, 0.067, 2.05*units.inch, 0.150],
    "S4H":  [0.87*units.inch, 0.051, 4.90*units.inch, 0.114],
    "S5L":  [0.12*units.inch, 0.100, 1.20*units.inch, 0.200],
    "S5M":  [0.34*units.inch, 0.083, 2.27*units.inch, 0.167],
    "S5H":  [1.09*units.inch, 0.063, 5.45*units.inch, 0.127],
    "C1L":  [0.10*units.inch, 0.062, 1.47*units.inch, 0.187],
    "C1M":  [0.29*units.inch, 0.052, 2.88*units.inch, 0.156],
    "C1H":  [0.50*units.inch, 0.024, 3.77*units.inch, 0.073],
    "C2L":  [0.12*units.inch, 0.100, 1.50*units.inch, 0.250],
    "C2M":  [0.26*units.inch, 0.083, 2.16*units.inch, 0.208],
    "C2H":  [0.74*units.inch, 0.063, 4.59*units.inch, 0.159],
    "C3L":  [0.12*units.inch, 0.100, 1.35*units.inch, 0.225],
    "C3M":  [0.26*units.inch, 0.083, 1.95*units.inch, 0.188],
    "C3H":  [0.74*units.inch, 0.063, 4.13*units.inch, 0.143],
    "PC1":  [0.18*units.inch, 0.150, 1.80*units.inch, 0.300],
    "PC2L": [0.12*units.inch, 0.100, 1.20*units.inch, 0.200],
    "PC2M": [0.26*units.inch, 0.083, 1.73*units.inch, 0.167],
    "PC2H": [0.74*units.inch, 0.063, 3.67*units.inch, 0.127],
    "RM1L": [0.16*units.inch, 0.133, 1.60*units.inch, 0.267],
    "RM1M": [0.35*units.inch, 0.111, 2.31*units.inch, 0.222],
    "RM2L": [0.16*units.inch, 0.133, 1.60*units.inch, 0.267],
    "RM2M": [0.35*units.inch, 0.111, 2.31*units.inch, 0.222],
    "RM2H": [0.98*units.inch, 0.085, 4.90*units.inch, 0.169],
    "URML": [0.24*units.inch, 0.200, 2.40*units.inch, 0.400],
    "URMM": [0.27*units.inch, 0.111, 1.81*units.inch, 0.222],
    "URMH": [0.27*units.inch, 0.111, 1.81*units.inch, 0.222],
}, index=["D_y", "A_y", "D_u", "A_u"], dtype=float)

pre_code_cc_table = pd.DataFrame({
    "W1":   [0.24*units.inch, 0.200, 4.32 *units.inch, 0.600],
    "W2":   [0.16*units.inch, 0.100, 2.35 *units.inch, 0.250],
    "S1L":  [0.15*units.inch, 0.062, 2.75 *units.inch, 0.187],
    "S1M":  [0.44*units.inch, 0.039, 5.33 *units.inch, 0.117],
    "S1H":  [1.16*units.inch, 0.024, 10.48*units.inch, 0.073],
    "S2L":  [0.16*units.inch, 0.100, 1.88 *units.inch, 0.200],
    "S2M":  [0.61*units.inch, 0.083, 4.85 *units.inch, 0.167],
    "S2H":  [1.94*units.inch, 0.063, 11.62*units.inch, 0.127],
    "S3":   [0.16*units.inch, 0.100, 1.88 *units.inch, 0.200],
    "S4L":  [0.10*units.inch, 0.080, 1.30 *units.inch, 0.180],
    "S4M":  [0.27*units.inch, 0.067, 2.46 *units.inch, 0.150],
    "S4H":  [0.87*units.inch, 0.051, 5.88 *units.inch, 0.114],
    "S5L":  [0.12*units.inch, 0.100, 1.20 *units.inch, 0.200],
    "S5M":  [0.34*units.inch, 0.083, 2.27 *units.inch, 0.167],
    "S5H":  [1.09*units.inch, 0.063, 5.45 *units.inch, 0.127],
    "C1L":  [0.10*units.inch, 0.062, 1.76 *units.inch, 0.187],
    "C1M":  [0.29*units.inch, 0.052, 3.46 *units.inch, 0.156],
    "C1H":  [0.50*units.inch, 0.024, 4.52 *units.inch, 0.073],
    "C2L":  [0.12*units.inch, 0.100, 1.80 *units.inch, 0.250],
    "C2M":  [0.26*units.inch, 0.083, 2.60 *units.inch, 0.208],
    "C2H":  [0.74*units.inch, 0.063, 5.51 *units.inch, 0.159],
    "C3L":  [0.12*units.inch, 0.100, 1.35 *units.inch, 0.225],
    "C3M":  [0.26*units.inch, 0.083, 1.95 *units.inch, 0.188],
    "C3H":  [0.74*units.inch, 0.063, 4.13 *units.inch, 0.143],
    "PC1":  [0.18*units.inch, 0.150, 2.16 *units.inch, 0.300],
    "PC2L": [0.12*units.inch, 0.100, 1.44 *units.inch, 0.200],
    "PC2M": [0.26*units.inch, 0.083, 2.08 *units.inch, 0.167],
    "PC2H": [0.74*units.inch, 0.063, 4.41 *units.inch, 0.127],
    "RM1L": [0.16*units.inch, 0.133, 1.92 *units.inch, 0.267],
    "RM1M": [0.35*units.inch, 0.111, 2.77 *units.inch, 0.222],
    "RM2L": [0.16*units.inch, 0.133, 1.92 *units.inch, 0.267],
    "RM2M": [0.35*units.inch, 0.111, 2.77 *units.inch, 0.222],
    "RM2H": [0.98*units.inch, 0.085, 5.88 *units.inch, 0.169],
    "URML": [0.24*units.inch, 0.200, 2.40 *units.inch, 0.400],
    "URMM": [0.27*units.inch, 0.111, 1.81 *units.inch, 0.222],
    "URMH": [0.27*units.inch, 0.111, 1.81 *units.inch, 0.222],
}, index=["D_y", "A_y", "D_u", "A_u"], dtype=float)

HAZUS_CAPACITY_CURVE_TABLE = pd.concat({
    "high-code": high_code_cc_table, 
    "moderate-code": moderate_code_cc_table, 
    "low-code": low_code_cc_table, 
    "pre-code": pre_code_cc_table}, axis=1)