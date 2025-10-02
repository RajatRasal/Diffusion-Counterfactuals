from typing import Dict, Tuple

from counterfactuals.scm.base import Mechanism
from counterfactuals.scm.image_mechanism.base import ImageMechanism


class SCM:

    def __init__(self, image_mechanism: Mechanism, numerical_mechanisms: Mechanism):
        self.image_mechanism = image_mechanism
        self.numerical_mechanisms = numerical_mechanisms 

    def abduct(self, cond: Dict) -> Dict:
        eps_numerical = self.numerical_mechanisms.abduct(cond)
        eps_image = self.image_mechanism.abduct(cond)
        eps = {**eps_numerical, **eps_image}
        return eps

    def predict(self, noise: Dict, interv: Dict) -> Dict:
        causes = self.numerical_mechanisms.predict(noise, interv)
        cf_images = self.image_mechanism.predict(noise, causes)
        return {**causes, **cf_images}

    def counterfactual(self, obs: Dict, interv: Dict) -> Dict:
        # total effect counterfactual
        return self.predict(self.abduct(obs), interv)
    
    def counterfactual_effects(self, obs: Dict, interv: Dict) -> Dict: #, target_variable: str) -> Dict:
        recon = self.counterfactual(obs, obs)

        noise = self.abduct(obs)
        te_parents = self.numerical_mechanisms.predict(noise, interv)
        te = self.image_mechanism.predict(noise, te_parents)

        # parents = self.numerical_mechanisms.predict(noise, obs)
        # de_parents = {
        #     k: te_parents[k] if k in interv else parents[k]
        #     for k, _ in obs.items() if k != "image"
        # }
        # de_parents = self.numerical_mechanisms.concat_metadata(de_parents)
        # # print({k: v.cpu() for k, v in sorted(de_parents.items()) if k != "image"})
        # de = self.image_mechanism.predict(noise, de_parents)

        # ide_parents = {
        #     k: te_parents[k] if k not in interv else parents[k]
        #     for k, _ in obs.items() if k != "image"
        # }
        # ide_parents = self.numerical_mechanisms.concat_metadata(ide_parents)
        # # print({k: v.cpu() for k, v in sorted(ide_parents.items()) if k != "image"})
        # ide = self.image_mechanism.predict(noise, ide_parents)

        # WRONG
        # # abduction
        # cf_parents = self.numerical_mechanisms.predict(noise, interv)

        # # direct effect
        # cf_parents_de = self.numerical_mechanisms.predict(eps, obs | interv)
        # print({k: v.cpu() for k, v in sorted(cf_parents_de.items()) if k != "image"})
        # direct_effect = self.image_mechanism.predict(eps, cf_parents_de)
        # # total effect
        # total_effect = self.image_mechanism.predict(eps, cf_parents)
        # print({k: v.cpu() for k, v in sorted(cf_parents.items()) if k != "image"})
        # cf_parents = self.numerical_mechanisms.predict(eps, interv)
        # # indirect effect
        # cf_parents_ide = cf_parents | {k: obs[k] for k, _ in interv.items()}
        # print({k: v.cpu() for k, v in sorted(cf_parents_ide.items()) if k != "image"})
        # indirect_effect = self.image_mechanism.predict(eps, cf_parents_ide)

        # direct_effect = self.counterfactual(obs, interv)
        # cf_parents = {
        #     k: v if k != target_variable else obs[target_variable]
        #     for k, v in direct_effect.items()
        # }
        # indirect_effect = self.counterfactual(cf_parents, obs)
        # total_effect = self.counterfactual(cf_parents, cf_parents)

        # return {"recon": recon, "de": direct_effect, "ie": indirect_effect, "te": total_effect}
        return {
            "recon": recon,
            # "de": de["image"],
            # "de_parents": de_parents,
            # "ie": ide["image"],
            # "ie_parents": ide_parents,
            "te": te["image"],
            "te_parents": te_parents,
        }
