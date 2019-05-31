

"""
(C) Asaph Zylbertal 01.03.2015, HUJI, Jerusalem, Israel
Run the mitral cell used in the article and produce example figures (Fig 4, Fig 5A-C, Fig 6C)
"""


import neuron
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt

def draw_model (mitral_mod):
    inst_dat={}
    
    inst_dat['v']=[]
    inst_dat['t']=[]

    # run model until steady state is reached, save in mitral_mod.steady    
    mitral_mod.init_steady_state(mitral_mod.soma(0.5), min_slope=5e-9, init_run_chunk=2000.)        
       
    
    # run model for short current injections    
    for amp in [-0.06, -0.03, 0.03, 0.06, 0.1, 0.15, 0.2, 0.35]:
    
        mitral_mod.steady.restore()            
        mitral_mod.init_square_stim(amp)
        mitral_mod.init_recording(mitral_mod.soma(0.5)) 
        mitral_mod.run_model()
    
        outv=np.array(mitral_mod.rec_v)
        outt=np.array(mitral_mod.rec_t)
        mitral_mod.stop_recording()            
        inst_dat['v'].append(outv)
        inst_dat['t'].append(outt)

    del mitral_mod.steady 
    del mitral_mod.stim
    

    ######### short current injections figure (article Fig 4) ##########
    fig=plt.figure(figsize=(5.5, 12))
    pas=fig.add_subplot(7, 1, 1)
    pas.plot(inst_dat['t'][0]/1000., inst_dat['v'][0],'g')
    pas.plot(inst_dat['t'][1]/1000., inst_dat['v'][1],'g')

    plt.tick_params(axis='x', which='both', bottom='on', labelbottom='off')
    
    plt.ylim(-75., -30.)
    plt.title('Passive response')
    plt.ylabel('Vm (mV)')
    
        
    mod2=fig.add_subplot(7, 1, 2)
    mod2.plot(inst_dat['t'][2], inst_dat['v'][2],'g')
    plt.ylim(-75., 60.)
    plt.title('I=30pA')
    plt.ylabel('Vm (mV)')
    
    plt.tick_params(axis='x', which='both', bottom='on', labelbottom='off')
        
    mod3=fig.add_subplot(7, 1, 3)
    mod3.plot(inst_dat['t'][3], inst_dat['v'][3],'g')
    plt.ylim(-75., 60.)
    plt.title('I=60pA')
    plt.ylabel('Vm (mV)')
    
    plt.tick_params(axis='x', which='both', bottom='on', labelbottom='off')

    mod4=fig.add_subplot(7, 1, 4)
    mod4.plot(inst_dat['t'][4], inst_dat['v'][4],'g')
    plt.ylim(-75., 60.)
    plt.title('I=100pA')
    plt.ylabel('Vm (mV)')
    
    plt.tick_params(axis='x', which='both', bottom='on', labelbottom='off')

    mod5=fig.add_subplot(7, 1, 5)
    mod5.plot(inst_dat['t'][5], inst_dat['v'][5],'g')
    plt.ylim(-75., 60.)
    plt.title('I=150pA')
    plt.ylabel('Vm (mV)')
    
    plt.tick_params(axis='x', which='both', bottom='on', labelbottom='off')

    mod6=fig.add_subplot(7, 1, 6)
    mod6.plot(inst_dat['t'][6], inst_dat['v'][6],'g')
    plt.ylim(-75., 60.)
    plt.title('I=200pA')
    plt.ylabel('Vm (mV)')
    
    plt.tick_params(axis='x', which='both', bottom='on', labelbottom='off')

    mod7=fig.add_subplot(7, 1, 7)
    mod7.plot(inst_dat['t'][7], inst_dat['v'][7],'g')
    plt.ylim(-75., 60.)
    plt.title('I=350pA')
    plt.xlabel('t (s)')
    plt.ylabel('Vm (mV)')
        
    
    ################



    four_spikes_frame_time=20.
    trains_frame_time=80.
    four_spikes_sim_time=15000
    trains_sim_time=60000
    
    # record "fluorescence" from tuft #1
    recording_comp=mitral_mod.tuft1(0.5)
    
    inst_dat={}            
    inst_dat['f']=[]
    inst_dat['v']=[]
    inst_dat['t']=[]
    inst_dat['i']=[]
    
    init_run_time=1500000
    

    ######### pump and exchanger current figure (article figure 6C) #########

    plt.figure()
    
    ca_pump_current=mitral_mod.init_vec_recording(mitral_mod.tuft1(0.5)._ref_ica_pmp_cadp)
    ca_ncx_rate=mitral_mod.init_vec_recording(mitral_mod.tuft1(0.5)._ref_rate_ncx)
    

    
    for freq in [1., 15., 30.]:

        if freq==1.:            
            # when the frequency is 1Hz (only four spikes) clamp to -55mV and inject -0.03nA during IC epoch (like the real experiment)       
            (t, ih, fl, v)=hybrid_clamp(mitral_mod,init_run_time, 3800-140,4400,4000,4000,freq/1000, 1,7.,four_spikes_sim_time,-55,6,-0.03, return_fluor=True, fluor_comp=recording_comp)
        else:
                
            # when the frequency is 15Hz or 30Hz clamp to -70mV and inject -0.06nA during IC epoch (like the real experiment)       
            (t, ih, fl, v)=hybrid_clamp(mitral_mod,init_run_time, 3800-140,4400,4000,4000,freq/1000, 1,7.,trains_sim_time,-70,6,-0.06, return_fluor=True, fluor_comp=recording_comp)
            
        
        outt=np.array(t)
        outf=np.array(fl)
        outv=np.array(v)
        
        if not freq==1:
            outi=np.array(ih)
            inst_dat['i'].append(outi)
            
        inst_dat['t'].append(outt)
        inst_dat['f'].append(outf)
        inst_dat['v'].append(outv)
        measure_frame_time=trains_frame_time/100
        
        if freq>1:
            interp_block_ca_pump_current=np.interp(np.arange(0,trains_sim_time+init_run_time,measure_frame_time), outt, np.array(ca_pump_current))[(init_run_time/measure_frame_time):]
            interp_block_ca_ncx_current=-2*np.interp(np.arange(0,trains_sim_time+init_run_time,measure_frame_time), outt, np.array(ca_ncx_rate))[(init_run_time/measure_frame_time):]

            measures_t=np.arange(0, trains_sim_time, measure_frame_time)

            if freq==30:  

                plt.plot(measures_t*1e-3 , interp_block_ca_pump_current*1e3, 'r')
                plt.plot(measures_t*1e-3 , interp_block_ca_ncx_current*1e3, 'g')
                plt.plot(measures_t*1e-3 , (interp_block_ca_ncx_current+interp_block_ca_pump_current)*1e3, 'b')
                plt.plot([0, 50], [0, 0], '--k')
                plt.xlim(0, 50)
                plt.ylim(-1., 1.)
                plt.title('Ca2+ pump and exchanger currents')
                plt.legend(('Ca2+ pump', 'Na+ - Ca2+ exchanger', 'Total'))
                plt.xlabel('t (sec)')
                plt.ylabel(r'$\mu A/cm^2$')

                
                
    
    
    ########## four spikes fluorescence figure (article figure 5A) #############
    
    plt.figure()

    interp_block=interp_f_result(inst_dat, four_spikes_sim_time, init_run_time, four_spikes_frame_time, 0, 0, mitral_mod.params['filt_order'], mitral_mod.params['time_shift'])
    plt.plot(np.arange(0,four_spikes_sim_time,four_spikes_frame_time)*1e-3, interp_block, 'b', linewidth=2.0)
    plt.xlabel('t (sec)')
    plt.ylabel('df/f')
    plt.title('Tuft fluorescence - 1Hz spiking')

    ########### 15Hz and 30Hz fluorescence and current figure (article figure 5B-C) ##########
    fig2=plt.figure(figsize=(14, 5))

    subfig=[0]*2
    subfig[0]=fig2.add_subplot(1, 2, 1)
    interp_block=interp_f_result(inst_dat, trains_sim_time, init_run_time, trains_frame_time, 1, 1, mitral_mod.params['filt_order'], mitral_mod.params['time_shift'])
    subfig[0].plot(np.arange(0, trains_sim_time, trains_frame_time)/1000., filtfilt(np.ones(5)/5, [1], interp_block), 'g', linewidth=2.0)
    
    interp_block=interp_f_result(inst_dat, trains_sim_time, init_run_time, trains_frame_time, 2, 2, mitral_mod.params['filt_order'], mitral_mod.params['time_shift'])
    subfig[0].plot(np.arange(0, trains_sim_time, trains_frame_time)/1000., filtfilt(np.ones(5)/5, [1], interp_block), 'r', linewidth=2.0)

    subfig[0].legend(('15Hz', '30Hz'))
    plt.xlabel('t (sec)')
    plt.ylabel('df/f')
    plt.title('Tuft fluorescence')
    
    subfig[1]=fig2.add_subplot(1, 2, 2)
    interp_block= interp_i_result(inst_dat, trains_sim_time, init_run_time, trains_frame_time, 1, 0)
    subfig[1].plot(np.arange(0, trains_sim_time, trains_frame_time)/1000., interp_block, 'g', linewidth=2.0)

    interp_block=interp_i_result(inst_dat, trains_sim_time, init_run_time, trains_frame_time, 2, 1)
    subfig[1].plot(np.arange(0, trains_sim_time, trains_frame_time)/1000., interp_block, 'r', linewidth=2.0)

    subfig[1].legend(('15Hz', '30Hz'))
    plt.xlabel('t (sec)')
    plt.ylabel('I (nC)')
    plt.title('Current recorded in the soma')

    plt.show()

 
def hybrid_clamp(inst, init_run_time, ic_delay, ic_duration, train_delay, train_duration, freq, amp, pulse_duration, sim_time, v_clamp, rs, dc=0.0, return_fluor=False, clamp_vec=None, clamp_t=None, fluor_comp=None):

    """
    run hybrid clamp experiment in the model:
    ----------------------------------------------
    inst - model cell instance
    init_run_time - initialization run time (how long to run to achieve steady state values in all state variables)
    ic_delay - delay before switch to from VC to IC
    ic_duration - duration of IC epoch
    train_delay - delay before pulse train injection
    train_duration - duration of the injected pulse train
    freq - pulse frequency
    amp - pulse amplitude
    pulse_duration - duration of each pulse in the train
    sim_tim - simulation duration (after initialization run)
    v_clamp - voltage to clamp to during VC
    rs - series resistance
    dc - DC current injection during IC epoch
    return_fluor (boolean) - should the function return simulated fluorescence data?
    clamp_vec - vector of command voltage (if not constant)
    clamp_t - time stamps for clamp_vec
    fluo_comp - compartment to read simulated fluorescence from (default=soma)
    """
    
    

    
    if fluor_comp==None:
        fluor_comp=inst.soma(0.5)
        
    t=inst.init_vec_recording(neuron.h._ref_t)
    
    hc=neuron.h.hybrid(inst.soma(0.5))
    i_hybrid=inst.init_vec_recording(hc._ref_i)
    v=inst.init_vec_recording(inst.tuft1(0.5)._ref_v)
    if return_fluor:    
        fl=inst.init_vec_recording(fluor_comp.cadp._ref_f)
    inst.init_train_stim(train_delay+init_run_time, train_duration, freq, pulse_duration, amp, 0.0, limit_dc=True)
    dcstim=neuron.h.IClamp(inst.soma(0.5))
    dcstim.delay=ic_delay+init_run_time
    dcstim.dur=ic_duration
    dcstim.amp=dc
    
    hc.rs=rs
    hc.delay=ic_delay+init_run_time
    hc.dur=ic_duration
    hc.tot_dur=sim_time+init_run_time
    
    if clamp_vec==None:
        hc.vc_amp=v_clamp
    else:
        stimv_vec=neuron.h.Vector(clamp_vec)
        t_vec=neuron.h.Vector(clamp_t)
        stimv_vec.play(hc._ref_vc_amp, t_vec)        

    
    if inst.cv.active()==1:
        inst.cv.re_init()      
    
    neuron.h.finitialize(v_clamp)
    neuron.h.fcurrent()
    neuron.run(sim_time+init_run_time)
    del inst.stim
    del hc
    del dcstim
    
    if return_fluor:
        return (t, i_hybrid, fl, v)
    else:
        return (t, i_hybrid,v)
        
def interp_f_result(inst_dat, sim_time, init_time, frame_time, t_num, f_num, filt_order, time_shift):

    """
    Interpolate and process fluorescence result
    """    
    
    interp_block=np.interp(np.arange(0,sim_time+init_time,frame_time), inst_dat['t'][t_num], inst_dat['f'][f_num])[(init_time/frame_time):]
    interp_block=df_f(interp_block)
    interp_block=filtfilt(np.ones(filt_order)/filt_order, [1], interp_block)
    origin_len=len(interp_block)        
    interp_block=np.concatenate([np.zeros(time_shift/frame_time), interp_block])
    interp_block=interp_block[0:origin_len]
    return interp_block    

def interp_i_result(inst_dat, sim_time, init_time, frame_time, t_num, i_num):

    """
    Interpolate current result
    """
    
    interp_block=np.interp(np.arange(0,sim_time+init_time,frame_time), inst_dat['t'][t_num], inst_dat['i'][i_num])[(init_time/frame_time):]
    return interp_block-np.mean(interp_block[0:3000/frame_time])

def df_f(f_vec):
    f_rest=np.mean(f_vec[0:10])
    return (f_vec-f_rest)/f_rest
    
