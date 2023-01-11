import kfp
from kfp import dsl
import kfp.components as components
                                           

@dsl.pipeline(name='Docker test', description='Applies Decision Tree and Logistic Regression for classification problem.')
def first_pipeline():

    # Loads the yaml manifest for each component
    getdata = kfp.components.load_component_from_file('getdata/getdata.yaml')
    reshapedata = kfp.components.load_component_from_file('reshapedata/reshapedata.yaml')
    modelbuilding = kfp.components.load_component_from_file('modelbuilding/modelbuilding.yaml')
    kserve_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/'
                                                'master/components/kserve/component.yaml') 
 
    # pipeline steps
    step1 = getdata()
    step2 = reshapedata()
    step2.after(step1)

    step3 = modelbuilding()
    step3.after(step2)


    kserve_op(action='apply',
            model_name='tensorflow-sample',
            model_uri='s3://mlpipeline/mnistdocker/models/detect-digits/',
            namespace='kubeflow-user-example-com',
            framework='tensorflow',
            service_account='sa-minio-kserve').after(step3)


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(first_pipeline, 'dockerten1Pipeline.yaml')
    # kfp.Client().create_run_from_pipeline_func(basic_pipeline, arguments={})