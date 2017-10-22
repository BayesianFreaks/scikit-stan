from typing import List
from typing import Dict

from jinja2 import Template

from sandbox.jinja_stan.stan import StanCodeComponent


class StanCodeMixin:

    def generate_stan_code(self,
                           data=None,
                           params=None,
                           transformed_params=None,
                           model=None):
        element_dict = {
            StanCodeComponent.DATA: data,
            StanCodeComponent.PARAMS: params,
            StanCodeComponent.TRANSFORMED_PARAMETERS_TEMPLATE: transformed_params,
            StanCodeComponent.MODEL: model,
        }

        stan_code = ''
        for component, elements in element_dict.items():
            if elements is not None:
                stan_code += component.template

        return self._render(
            stan_code,
            {k.variable_name: v for k, v in element_dict.items()}
        )

    def _render(self, code: str, elements: Dict[str, List[str]]):
        template = Template(code)
        return template.reder(**elements)
