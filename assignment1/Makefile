

# Commands for McCulloch-Pitts exercise
ayuda_mc:
	@echo "======> AYUDA PARA EL PROGRAMA: RED DE MCCULLOCH-PITTS <======"
	@echo "\t ·Nombre del programa ejecutable de Python3: ex_mcculloch_pitts.py"
	@echo "\t ·Posibles parámetros de entrada:"
	@echo "\t\t - (obligatorio) ruta fichero de lectura donde se especifican las entradas del ejercicio"
	@echo "\t ·EJEMPLO DE USO: python3 ex_mcculloch_pitts.py test_files/mccPitts_ex.txt"
	@echo "=============================================================================="

compila_mc:
	@echo "Python es un lenguaje interpretado"
	@echo "De todas formas, asegúrese de satisfacer los prerrequisitos:"
	@make requirements

ejecuta_mc:
	@python3 ex_mcculloch_pitts.py test_files/mccPitts_ex.txt


# Commands for Perceptron exercise


# Commands for Adaline exercise


# General useful commands
requirements:
	@echo "===> LIBRERÍAS NO-ESTÁNDAR QUE DEBERÍAN INSTALARSE <==="
	@echo "\t numpy: librería de cálculo matemático eficiente para Python"
	@echo "\t tabulate: librería de formateo de tablas para Python"
	@echo "==> Si desea instalar dichas librerías, puede ejecutar \'make install_requirements\'"

install_requirements:
	@pip3 install numpy
	@pip3 install tabulate
