from torchdiffeq import odeint_adjoint,odeint
import libsbml
import torch
import numpy as np
from torch import nn

def load_sbml_model(file_path):
    """loading model from file_path"""
    reader=libsbml.SBMLReader()
    document=reader.readSBML(file_path)
    print("Number of internal inconsistencies",document.checkInternalConsistency())

    model=document.getModel()
    print("Number of species:",model.getNumSpecies())
    print("Number of reactions:",model.getNumReactions())
    print("Number of constant boundary metabolites",model.getNumSpeciesWithBoundaryCondition())
    return model

def get_initial_conditions(model):
    """Retrieves the species initial concentrations 
    from the SBML model. If a species is a constant boundary condition, 
    then it should be passed as a parameter instead of an initial condition,
    since it does not have a rate law"""
    species=model.getListOfSpecies()
    initial_concentration_dict={}
    for i in range(len(species)):
        specimen=species[i]

        #there are also non-stationary boundary conditions, deal with this later. 
        if specimen.getConstant() and specimen.getBoundaryCondition():
            continue
            # print("Constant Boundary Specimen ",specimen.id)
        else:   
            initial_concentration_dict[species[i].id]=specimen.initial_concentration
    return initial_concentration_dict

def get_model_parameters(model):
    """retrieves parameters from the loaded sbml model. 
    These will be passed to the Torch Kinetic mechanism class. And that will be passed to torch_sbml_kinetic_model (similar to the other structures)
    
    Values of dictionary keys that are nan are later checked and filled in the get_parameters_for_evaluation function()


    Returns:
    - parameter_dict (learnable)
    - compartment_dict (nont learnable)
    - boundary_dict (not learnable)
    """
    #Global parameters in sbml
    parameter_dict={}
    boundary_dict={}
    compartment_dict={}
    params=model.getListOfParameters()
    for i in range(len(params)):
        parameter_dict[params[i].id]=params[i].value
    compartments=model.getListOfCompartments()
    
    #Local parameters (if existing, not every sbml model has locally (reaction defined) parameters)
    for i in range(len(model.reactions)):
        reaction=model.reactions[i]
        id=reaction.id
        r=reaction.getKineticLaw()
        for i in range(len(r.getListOfParameters())):
            key=id+"_"+r.getListOfParameters()[i].id
            value=r.getListOfParameters()[i].value
            parameter_dict[key]=value

    #compartments are added to the parameter_dict because they are need in the evaluation. Shouldnt be learnable
    for i in range(len(compartments)):
        compartment_dict[compartments[i].id]=compartments[i].size

    #If species is a boundary, this should be added to the parameter_dict, since it does
    #not have a rate law
    species=model.getListOfSpecies()
    for i in range(len(species)):
        specimen=species[i]
        if specimen.getConstant() and specimen.getBoundaryCondition():
            boundary_dict[specimen.id]=specimen.initial_concentration
    return parameter_dict,boundary_dict,compartment_dict

#this function might not be required.
def get_reaction_species(reaction):
    """Retrieves the substrates, products, and modifiers from sbml format. 
     These will be passed to the Torch Kinetic Model class. """
    sub=reaction.getListOfReactants()
    prod=reaction.getListOfProducts()
    mod=reaction.getListOfModifiers()
    
    substrates=[]
    products=[]
    modifiers=[]
    for i in range(len(sub)):
        substrates.append(sub[i].species)
    for i in range(len(prod)):
        products.append(prod[i].species)
    for i in range(len(mod)):
        modifiers.append(mod[i].species)

    species=substrates+products+modifiers
    return species

def get_string_expression(reaction):
    #retrieves the kinetic rate law from the sbml model
    kinetic_law=reaction.getKineticLaw()  
    klaw_math=kinetic_law.math
    string_rate_law=libsbml.formulaToString(klaw_math)
    # here we sometimes need to add exceptions. For example, to evaluate tanh, we need to replace it with torch.Tanh
    string_rate_law=string_rate_law.replace("^","**")
    return string_rate_law

def get_stoichiometry_for_species(model, species_id):
    #Thanks chatGPT
    # Get the species by ID
    species = model.getSpecies(species_id)

    if species is None:
        print(f"Species with ID '{species_id}' not found.")
        return
    
    # Dictionary to store the reactions and their stoichiometries
    reactions_info = {}

    # Iterate through all reactions to find the specified species
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)

        # Check for reactants
        for j in range(reaction.getNumReactants()):
            reactant = reaction.getReactant(j)
            if species.getId() == reactant.getSpecies():
                stoichiometry = -1 * reactant.getStoichiometry()
                reactions_info[reaction.getId()] = stoichiometry

        # Check for products
        for j in range(reaction.getNumProducts()):
            product = reaction.getProduct(j)
            if species.getId() == product.getSpecies():
                stoichiometry = product.getStoichiometry()
                reactions_info[reaction.getId()] = stoichiometry
    return reactions_info

def get_parameters_for_evaluation(reaction,parameter_dict):
    """retrieve parameters that are used for evaluating the expression"""
    #retrieve parameters
    id=reaction.id
    
    kinetic_law=reaction.getKineticLaw()  
    klaw_math=kinetic_law.math
    string_rate_law=libsbml.formulaToString(klaw_math)
# ey=id+"_"+r.getListOfParameters()[i].id
    #parse string
    operators_to_remove = ["+","-" ,"*","/","^","(",")",","]
    temp=string_rate_law
    for i in operators_to_remove:
        temp=temp.replace(i,"~") #replace everywhere in between with an extra space, because if you remove ( then tan and n will be together.
    temp=temp.replace(" ","")
    splitted_rate_law=temp.split("~")     
    keys=[]
    for i in splitted_rate_law:
        i=i.strip()
        
        if i in parameter_dict:
            keys.append(i)
        k=reaction.id+"_"+i
        if k in parameter_dict:
            keys.append(k)
        #perhaps we need to add substrates, products, and modifiers later    
    #add local parameters that are necessary for flux calculations
    local_parameter_dict={key: parameter_dict[key] for key in keys}
    #replace key value with a different key
    local_parameter_dict={key.replace(reaction.id+"_",""): value for key, value in local_parameter_dict.items()}
    return local_parameter_dict

def evaluate_piecewise_expression(expression,t):
    """Evaluate a piecewise expression. This function is very specific to
    the yeast model ss model, might be not important."""
    expression=expression.replace("piecewise(","")
    # splitted_expression=expression.split("), ")
    expression=expression.split(", lt")
    values=[]
    condition_values=[]
    for i in expression:
        i=i.replace("(","")
        i=i.replace(")","")
        i=i.split(",")
        if len(i)==1:
            values.append(float(i[0]))
        else:
            values.append(float(i[2]))
            condition_values.append(float(i[1]))
    condition_values.insert(0,0)
    for i in range(len(condition_values)-1):
        if t>=condition_values[i] and t<condition_values[i+1]:
            expression_eval=values[i]
        elif t>=condition_values[i+1]:
            expression_eval=values[i+1]
    return torch.Tensor([expression_eval])

def fill_in_assignment_rules(model,subset_parameters):
    """for the dictionary specific for a flux, fill in any value that is defined by some assignment rule
    Now this function only takes piecewise functions, but for other expressions it might be enough to do an evaluation of the rule. Worries for later.
    """
    t=subset_parameters['t']
    for i in subset_parameters.keys():
        rule=model.getRuleByVariable(i)
        if rule!=None:
            string_assignment_law=rule.getFormula()
            value=evaluate_piecewise_expression(string_assignment_law,t)
            subset_parameters[i]=value
    return subset_parameters


# class torch_SBML_rate_law(torch.nn.Module):  
#     def __init__(self,
#             sbml_reaction,
#             parameter_dict,
#             metabolite_names):
#         super(torch_SBML_rate_law, self).__init__()

#         self.species=get_reaction_species(sbml_reaction)
#         self.string_rate_law=get_string_expression(sbml_reaction)
#         local_parameters=get_parameters_for_evaluation(reaction=sbml_reaction,
#                                                       parameter_dict=parameter_dict)
        

#         self.parameter_dict=torch.nn.ParameterDict(local_parameters)
#         # print(self.parameter_dict)
#         self.metabolite_names=metabolite_names
#         self.evaluation=compile(self.string_rate_law,"<string>","eval")

#     def calculate(self,concentrations):
#         ## This will be done slightly different. Instead of subsetting substrates, products, modifiers
#         #Add all of them and let the eval function sort it out. This makes it simpler to code
#         # However, then we have to pass everything in the forward pass
#         temp_dict=dict(zip(self.metabolite_names,concentrations))
#         m={i:torch.Tensor([temp_dict[i]]) for i in self.species if i in self.metabolite_names} #this is buggy?
#         eval_dict=self.parameter_dict
#         eval_dict={**eval_dict,**m}
#         v=eval(self.evaluation,eval_dict)

#         return torch.Tensor([v])




# class torch_SBML_kinetic_model(torch.nn.Module):
#     def __init__(self,
#                  model,
#                  parameter_dict):
#         super(torch_SBML_kinetic_model,self).__init__()
#         self.metabolite_names=list(get_initial_conditions(model).keys())

#         #sanity check: model parameter keys should all be unique
#         if len(parameter_dict)!=len(np.unique(list(parameter_dict.keys()))):
#             print("number of parameters not unique")
#         # self.parameter_dict=torch.nn.ParameterDict(parameter_dict)
#         print(parameter_dict)
        
#         #set up fluxes
#         fluxes={}
#         for i in range(model.getNumReactions()):
#             v=torch_SBML_rate_law(model.reactions[i],parameter_dict=self.parameter_dict,metabolite_names=self.metabolite_names)
#             fluxes[model.reactions[i].id]=v
#         self.fluxes=fluxes

#         #get stoichiometric info
#         self.stoich={}
#         for i in self.metabolite_names:
#             temp=get_stoichiometry_for_species(model,i)
#             self.stoich[i]=temp
#         if len(self.stoich)!=len(self.metabolite_names):
#             print("mismatch between metabolites and rows (metabolites) of stoichiometry")

#     def calculate_fluxes(self,_,concentrations):
#         for i in self.fluxes:
#             self.fluxes[i].parameter_dict["t"]=torch.Tensor([_])
#             # self.fluxes[i].evaluation_dictionary["n"]=torch.Tensor([485])
#             self.fluxes[i].value=self.fluxes[i].calculate(concentrations)
               

#     def forward(self,_,conc_in):

#         self.calculate_fluxes(_,conc_in)
#         dXdt=torch.Tensor([])
#         for k,i in enumerate(self.metabolite_names):
#             if len(self.stoich[i])!=0: #check whether stoichiometry is not empty (e.g. it is a modifier)
#                 x=sum([self.fluxes[j].value*self.stoich[i][j] for j in self.stoich[i]])
#             else:
#                 x=torch.Tensor([0])
#             dXdt=torch.cat([dXdt,x],dim=0)
#         return dXdt
    

class torch_SBML_kinetic_model(torch.nn.Module):
    def __init__(self,
                 model,
                 fluxes): #metabolites might not be necessary.
        super(torch_SBML_kinetic_model,self).__init__()
        self.fluxes=nn.ParameterDict(fluxes)
        self.metabolite_names=list(get_initial_conditions(model).keys())
        n_parameters=len(list(self.fluxes.parameters()))

        #get stoichiometric info
        self.stoich={}
        for i in self.metabolite_names:
            temp=get_stoichiometry_for_species(model,i)
            self.stoich[i]=temp
        if len(self.stoich)!=len(self.metabolite_names):
            print("mismatch between metabolites and rows (metabolites) of stoichiometry")



    def calculate_fluxes(self,_,concentrations):
        for i in self.fluxes:
            # self.fluxes[i].named_parameters["t"]=torch.Tensor([_])
            # self.fluxes[i].evaluation_dictionary["n"]=torch.Tensor([485])
            self.fluxes[i].value=self.fluxes[i].calculate(concentrations)
        # # self.parameter_dict=torch.nn.ParameterDict(parameter_dict)
        # print(parameter_dict)
    def forward(self,_,conc_in):
        self.calculate_fluxes(_,conc_in)
        dXdt=torch.Tensor([])
        for k,i in enumerate(self.metabolite_names):
            if len(self.stoich[i])!=0: #check whether stoichiometry is not empty (e.g. it is a modifier)
                x=sum([self.fluxes[j].value*self.stoich[i][j] for j in self.stoich[i]])
            else:
                x=torch.Tensor([0])
            dXdt=torch.cat([dXdt,x],dim=0)
        return dXdt