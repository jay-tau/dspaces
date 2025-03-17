def main():
    from writer import Writer

    test_reader = Writer(n_dims=2, dims=[2, 4], procs=[2, 2])
    test_reader.generate_data()


if __name__ == "__main__":
    main()
