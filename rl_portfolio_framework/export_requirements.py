import os

def export_conda_env(filename="rl_portfolio_env.yml", no_builds=False):
    """
    Exportiert die Conda-Umgebung in eine YAML-Datei.
    :param filename: Name der Exportdatei
    :param no_builds: Entfernt hardware-spezifische Informationen, wenn True
    """
    command = f"conda env export > {filename}"
    if no_builds:
        command = f"conda env export --no-builds > {filename}"
    exit_code = os.system(command)
    if exit_code == 0:
        print(f"✅ Conda-Umgebung exportiert nach: {filename}")
    else:
        print(f"❌ Fehler beim Exportieren der Conda-Umgebung nach: {filename}")


def export_pip_requirements(filename="rl_portfolio_requirements.txt"):
    """
    Exportiert die Pip-Requirements in eine TXT-Datei.
    :param filename: Name der Exportdatei
    """
    command = f"pip freeze > {filename}"
    exit_code = os.system(command)
    if exit_code == 0:
        print(f"✅ Pip-Requirements exportiert nach: {filename}")
    else:
        print(f"❌ Fehler beim Exportieren der Pip-Requirements nach: {filename}")


def main():
    try:
        # Export der Conda-Umgebung mit hardware-spezifischen Informationen
        export_conda_env(filename="rl_portfolio_env_full.yml", no_builds=False)

        # Export der Conda-Umgebung ohne hardware-spezifische Informationen
        export_conda_env(filename="rl_portfolio_env.yml", no_builds=True)

        # Export der Pip-Requirements
        export_pip_requirements(filename="rl_portfolio_requirements_full.txt")

    except FileNotFoundError as e:
        print(e)
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler beim Ausführen eines Befehls: {e}")


if __name__ == "__main__":
    main()


# In der Konsole ausführen:
# $ python export_requirements.py


# pip freeze > rl_portfolio_requirements_full.txt
# conda env export > rl_portfolio_env_full.yml   
# conda env export --no-builds > rl_portfolio_env.yml