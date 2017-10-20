from enum import Enum


class StanCodeTemplate:

    DATA_TEMPLATE = '''
        data {
            {% for df in definitions  %}
                {{df}};
            {% endfor %}
        }
        '''

    PARAMETER_TEMPLATE = '''
        parameters {
            {% for param in params %}
                {{param}};
            {% endfor %}
        }
        '''

    TRANSFORMED_PARAMETERS_TEMPLATE = '''
        transformed parameters {
            {% for param in transformed_params %}
                {{param}};
            {% endfor %}
        }
        '''
    MODEL_TEMPLATE = '''
        model {
            {% for dist in distributions %}
                {{dist}};
            {% endfor %}
        }
        '''


class StanCodeComponent(Enum):

    def __init__(self, variable_name, template):
        self.variable_name = variable_name
        self.template = template

    DATA = ('definitions', StanCodeTemplate.DATA_TEMPLATE)
    PARAMS = ('params', StanCodeTemplate.PARAMETER_TEMPLATE)
    TRANSFORMED_PARAMS = ('transformed_params', StanCodeTemplate.TRANSFORMED_PARAMETERS_TEMPLATE)
    MODEL = ('distributions', StanCodeTemplate.MODEL_TEMPLATE)
