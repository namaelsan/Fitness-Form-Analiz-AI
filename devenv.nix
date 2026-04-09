{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # https://devenv.sh/basics/c
  env.GREET = "devenv";

  # Allow pip-installed binaries to find standard Nix libraries and the GPU drivers
  env.LD_LIBRARY_PATH =
    lib.makeLibraryPath (
      with pkgs;
      [
        stdenv.cc.cc.lib
        zlib
        glib
        xorg.libX11
        xorg.libXext
        xorg.libXrender
        libglvnd
      ]
    )
    + ":/run/opengl-driver/lib:/run/opengl-driver-32/lib";

  # https://devenv.sh/packages/
  packages = with pkgs; [
    # Add a C compiler for building some Python packages
    stdenv.cc.cc.lib
    # General dependencies often needed for Python/ML/computer vision on Nix
    zlib
    git
    libGL
    glib

    xorg.libX11 # for opencv webcam
    xorg.libXext
    xorg.libXrender
    libglvnd
    libsm
    libice

  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true;
    venv.requirements = ''
      mediapipe==0.10.14
      tensorflow==2.15.1
      jax==0.4.23
      jaxlib==0.4.23
      protobuf>=4.21,<5
      tkinter
      dearpygui
      numpy
      pandas
      opencv-python
      ultralytics
    '';
  };

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # https://devenv.sh/basics/
  enterShell = ''
    export QT_QPA_PLATFORM=xcb

    echo "Gym Form Analysis Environment Loaded"
    echo "Python: $(python --version)"
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
