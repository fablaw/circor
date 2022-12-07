reinstall_package:
	@pip uninstall -y circor || :
	@pip install -e .

run_preprocess_2D_and_save:
	python -c 'from circor.interface.main import wav_to_2D; wav_to_2D()'

run_preprocess_1D_and_save:
	python -c 'from circor.interface.main import wav_to_1D; wav_to_1D()'

streamlit:
	-@streamlit run circor/app/app.py

run_api:
	uvicorn circor.api.api:app --reload
