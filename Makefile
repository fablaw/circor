reinstall_package:
	@pip uninstall -y circor || :
	@pip install -e .

run_preprocess:
	python -c 'from circor.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from circor.interface.main import train; train()'

run_pred:
	python -c 'from circor.interface.main import pred; pred()'

run_evaluate:
	python -c 'from circor.interface.main import evaluate; evaluate()'

run_all: run_train run_evaluate run_pred

streamlit:
	-@streamlit run circor/app/app.py

run_api:
	uvicorn circor.api.api:app --reload
