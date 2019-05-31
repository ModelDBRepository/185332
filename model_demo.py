


import neuron, model_definition, model_run

neuron.load_mechanisms('./channels/')
mitral_mod=model_definition.full_mitral_neuron()
model_run.draw_model(mitral_mod)

