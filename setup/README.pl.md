# Instrukcja Instalacji

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/kzajac97/machine-vision/tree/main/setup/README.md)

## Instalacja

1. Zainstaluj Pythona 3.x ze strony oficjalnej: https://www.python.org/downloads/ (zalecana wersja 3.9+)
2. Zainstaluj pip, narzędzie do instalacji pakietów Pythona, wykonując poniższą komendę w terminalu: `python -m ensurepip --default-pip`
3. Zainstaluj virtualenv, narzędzie do tworzenia izolowanych środowisk Pythona, wykonując poniższą komendę w terminalu: `pip install virtualenv`
4. Utwórz nowe środowisko wirtualne dla swojego projektu, wykonując poniższą komendę w terminalu: `virtualenv env`
5. Aktywuj środowisko wirtualne, wykonując poniższą komendę w terminalu: `source env/bin/activate` na Linuxie lub `env\Scripts\activate.bat` na Windowsie.
6. Zainstaluj wymagane pakiety dla swojego projektu, wykonując poniższą komendę w terminalu: `pip install -r requirements.txt`

*Uwaga*: Uruchamianie Pythona z terminala ułatwia korzystanie z `venv`, aby upewnić się, że używana jest właściwa wersja.

## Użycie

1. Uruchom Jupyter Notebook, wykonując poniższą komendę w terminalu: `jupyter notebook` (lub uruchom `jupyter lab` dla bogatszego środowiska przeglądarki)
2. Otwórz plik notatnika w przeglądarce, klikając na dostarczony link w wyniku działania terminala (w środowisku Visual Studio Code otwórz plik `ipynb` i wklej URL w `Select Kernel` - `Select Another Kernel` - `Existing Jupyter Server`).
3. Uruchamiaj komórki notatnika, klikając przycisk "Run" lub naciskając "Shift + Enter" na klawiaturze.

# Git

## Rozpoczęcie pracy

1. Utwórz konto na GitHubie na stronie https://github.com/.
2. Zainstaluj Git na swoim komputerze, postępując zgodnie z instrukcjami na stronie https://git-scm.com/downloads.
3. Utwórz nowe repozytorium na GitHubie, klikając przycisk "New" na stronie głównej.
4. Sklonuj repozytorium na swój komputer, wykonując poniższą komendę w terminalu: `git clone <repository-url>`.
5. Utwórz nowy plik w repozytorium, wykonując poniższą komendę w terminalu: `touch <filename>`.
6. Dodaj plik do obszaru stage, wykonując poniższą komendę w terminalu: `git add <filename>`.
7. Zatwierdź zmiany, wykonując poniższą komendę w terminalu: `git commit -m "Wiadomość commita"`.
8. Wypchnij zmiany do zdalnego repozytorium, wykonując poniższą komendę w terminalu: `git push origin master`.

## Branching

Rozgałęzianie pozwala na stworzenie kopii twojego kodu i pracę nad nią niezależnie od głównej gałęzi.

1. Utwórz nową gałąź, wykonując poniższą komendę w terminalu: `git branch <branch-name>`.
2. Przełącz się na nową gałąź, wykonując poniższą komendę w terminalu: `git checkout <branch-name>`.
3. Dokonaj zmian w kodzie.
4. Dodaj i zatwierdź zmiany jak wcześniej.
5. Wypchnij zmiany do zdalnej gałęzi, wykonując poniższą komendę w terminalu: `git push origin <branch-name>`.