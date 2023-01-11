import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp import components

# kfserving_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/'
#                                                  'master/components/kubeflow/kfserving/component.yaml')
kserve_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/'
                                               'master/components/kserve/component.yaml')


@dsl.pipeline(
    name='KServe pipeline',
    description='A pipeline for KServe.'
)
def kservePipeline(
        action='apply',
        model_name='tensorflow-sample',
        model_uri='s3://mlpipeline/mnistdocker/models/detect-digits/',
        namespace='kubeflow-user-example-com',
        framework='tensorflow',
        service_account='sa-minio-kserve'):
    kserve_op(action=action,
            model_name=model_name,
            model_uri=model_uri,
            namespace=namespace,
            framework=framework,
            service_account=service_account)



if __name__ == '__main__':
    compiler.Compiler().compile(kservePipeline, 'pip.yaml')