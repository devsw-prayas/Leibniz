#pragma once

// A sexy ass header that focuses on compile time unrolling ;)

namespace Leibniz::Utils::Unrolled {
		template<size_t  Idx>
		using IndexT = std::integral_constant<size_t, Idx>;

	template<bool Condition, typename TrueFn, typename FalseFn>
	constexpr void staticIf(TrueFn&& u_Tfunc, FalseFn&& u_Ffunc) {
		if constexpr (Condition) u_Tfunc();
		else u_Ffunc();
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr bool staticAny(Fn&& u_Func) {
		bool result = false;
		[&] <size_t... I>(std::index_sequence<I...>) {
			((result |= u_Func(IndexT<Begin + I>{})), ...);
		}(std::make_index_sequence<End - Begin>{});
		return result;
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr bool staticAll(Fn&& u_Func) {
		bool result = true;
		[&] <size_t... I>(std::index_sequence<I...>) {
			((result &= u_Func(IndexT<Begin + I>{})), ...);
		}(std::make_index_sequence<End - Begin>{});
		return result;
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr size_t staticCount(Fn&& u_Func) {
		size_t count = 0;
		[&] <size_t... I>(std::index_sequence<I...>) {
			((count += u_Func(IndexT<Begin + I>{}) ? 1 : 0), ...);
		}(std::make_index_sequence<End - Begin>{});
		return count;
	}

	template<size_t Begin, size_t End, typename T, typename Fn>
	constexpr T	staticReduce(T init, Fn&& u_Func) {
		T value = init;
		[&] <size_t... I>(std::index_sequence<I...>) {
			((value = u_Func(value, IndexT<Begin + I>{})), ...);
		}(std::make_index_sequence<End - Begin>{});
		return value;
	}

	template<size_t Begin, size_t End, typename F>
	constexpr auto staticSum(F&& u_Func) {
		using T = decltype(u_Func(IndexT<Begin>{}));
		T result{};
		[&] <size_t... I>(std::index_sequence<I...>) {
			((result += u_Func(IndexT<Begin + I>{})), ...);
		}(std::make_index_sequence<End - Begin>{});

		return result;
	}

	template<size_t Begin, size_t End, typename F>
	constexpr auto staticMin(F&& u_Func) {
		auto value = u_Func(IndexT<Begin>{});
		[&] <size_t... I>(std::index_sequence<I...>) {
			((value = std::min(value, u_Func(IndexT<Begin + I>{}))), ...);
		}(std::make_index_sequence<End - Begin>{});
		return value;
	}

	template<size_t Begin, size_t End, typename F>
	constexpr auto staticMax(F&& u_Func) {
		auto value = u_Func(IndexT<Begin>{});
		[&] <size_t... I>(std::index_sequence<I...>) {
			((value = std::max(value, u_Func(IndexT<Begin + I>{}))), ...);
		}(std::make_index_sequence<End - Begin>{});
		return value;
	}

	template<typename...T>
	struct TypeList final {};

	template<typename...T, typename Fn>
	constexpr void staticForeachType(TypeList<T...>, Fn&& u_Func) {
		(u_Func.template operator() < T > (), ...);
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticFor(Fn&& u_Func) {
		[&] <size_t... I>(std::index_sequence<I...>) {
			(u_Func(IndexT<Begin + I>{}), ...);
		}(std::make_index_sequence<End - Begin>{});
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticForReverse(Fn&& u_Func) {
		[&] <size_t... I>(std::index_sequence<I...>) {
			(u_Func(IndexT<End - 1 - I>{}), ...);
		}(std::make_index_sequence<End - Begin>{});
	}

	template<size_t N, typename Fn>
	constexpr void staticRepeat(Fn&& u_Func) {
		staticFor<0, N>([&](auto) {
			u_Func();
						});
	}

	template<size_t Begin, size_t End, size_t Stride, typename Fn>
	constexpr void staticStride(Fn&& u_Func) {
		staticFor<0, (End - Begin + Stride - 1) / Stride>(
			[&](auto i) {
				constexpr size_t idx = Begin + i * Stride;

				if constexpr (idx < End)
					u_Func(IndexT<idx>{});
			});
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticPairwise(Fn&& u_Func) {
		staticFor<Begin, End - 1>(
			[&](auto i) {
				constexpr size_t a = i;
				constexpr size_t b = i + 1;

				u_Func(IndexT<a>{}, IndexT<b>{});
			});
	}

	template<size_t BeginA, size_t EndA, size_t BeginB, size_t EndB, typename Fn>
	constexpr void staticZip(Fn&& u_Func) {
		constexpr size_t Count = (EndA - BeginA < EndB - BeginB) ? (EndA - BeginA) : (EndB - BeginB);
		staticFor<0, Count>([&](auto i) {
			u_Func(IndexT<BeginA + i>{}, IndexT<BeginB + i>{}); });
	}

	template<size_t Begin, size_t End, size_t Chunk, typename Fn>
	constexpr void staticChunk(Fn&& u_Func) {
		staticStride<Begin, End, Chunk>([&](auto start) {
			u_Func(start, IndexT<Chunk>{}); });
	}

	template<size_t Begin, size_t End, size_t Window, typename Fn>
	constexpr void staticWindow(Fn&& u_Func) {
		staticFor<Begin, End - Window + 1>(
			[&](auto i) {
				u_Func(i);
			});
	}

	template<size_t Begin, size_t End, size_t TileSize, typename Fn>
	constexpr void staticTile(Fn&& u_Func) {
		staticStride<Begin, End, TileSize>(
			[&](auto tileStart) {
				u_Func(tileStart);
			});
	}

	template<typename Range, typename Fn>
	constexpr void staticRangeFor(Fn&& u_Func) {
		staticFor<Range::begin, Range::end>(
			[&](auto i) {
				u_Func(i);
			});
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticEnumerate(Fn&& u_Func) {
		staticFor<Begin, End>(
			[&](auto i) {
				constexpr size_t idx = i;
				u_Func(IndexT<idx>{});
			});
	}

	template<size_t Begin, size_t End, typename Mapper, typename Fn>
	constexpr void staticIndexMap(Fn&& u_Func) {
		staticFor<Begin, End>(
			[&](auto i) {
				constexpr size_t mapped = Mapper::template map<i>();
				u_Func(IndexT<mapped>{});
			});
	}

	template<size_t Rows, size_t Cols, typename Fn>
	constexpr void staticGrid(Fn&& u_Func) {
		staticFor<0, Rows>([&](auto r) {
			staticFor<0, Cols>([&](auto c) {
				u_Func(r, c); }); });
	}

	template<size_t X, size_t Y, typename Fn>
	constexpr void staticGrid2D(Fn&& u_Func) {
		staticFor<0, X>([&](auto x) {
			staticFor<0, Y>([&](auto y) {
				u_Func(x, y);
							});
						});
	}

	template<size_t X, size_t Y, size_t Z, typename Fn>
	constexpr void staticGrid3D(Fn&& u_Func) {
		staticFor<0, X>([&](auto x) {
			staticFor<0, Y>([&](auto y) {
				staticFor<0, Z>([&](auto z) {
					u_Func(x, y, z);
								});
							});
						});
	}

	template<size_t N, typename Fn>
	constexpr void staticCube(Fn&& u_Func) {
		staticGrid3D<N, N, N>(u_Func);
	}

	template<typename Fn, size_t... Dims>
	constexpr void staticHyperGrid(Fn&& u_Func) {
		constexpr size_t Rank = sizeof...(Dims);

		auto dims = std::array<size_t, Rank>{ Dims... };

		std::array<size_t, Rank> indices{};

		auto recurse = [&](auto&& self, size_t depth) -> void {
			if (depth == Rank) {
				u_Func(indices);
				return;
			}

			for (size_t i = 0; i < dims[depth]; ++i) {
				indices[depth] = i;
				self(self, depth + 1);
			}
			};

		recurse(recurse, 0);
	}

	template<size_t N, typename Fn>
	constexpr void staticDiagonal(Fn&& u_Func) {
		staticFor<0, 2 * N - 1>([&](auto d) {
			constexpr size_t diag = d;

			staticFor<0, N>([&](auto i) {
				constexpr size_t ii = i;
				constexpr size_t jj = diag - ii;

				if constexpr (jj < N)
					u_Func(IndexT<diag>{}, IndexT<ii>{}, IndexT<jj>{});
							});
								});			
	}

	template<size_t N, typename Fn>
	constexpr void staticTriangle(Fn&& u_Func) {
		staticFor<0, N>([&](auto i) {
			staticFor<0, i + 1>([&](auto j) {
				u_Func(i, j);
								});
						});
	}

	template<size_t N, typename Fn>
	constexpr void staticUpperTriangle(Fn&& u_Func) {
		staticFor<0, N>([&](auto i) {
			staticFor<i, N>([&](auto j) {
				u_Func(i, j);
							});
						});
	}

	template<size_t N, typename Fn>
	constexpr void staticLowerTriangle(Fn&& u_Func) {
		staticFor<0, N>([&](auto i) {
			staticFor<0, i + 1>([&](auto j) {
				u_Func(i, j);
								});
						});
	}

	template<size_t N, size_t Band, typename Fn>
	constexpr void staticBand(Fn&& u_Func) {
		staticFor<0, N>([&](auto i) {
			staticFor<0, N>([&](auto j) {
				constexpr size_t ii = i;
				constexpr size_t jj = j;

				if constexpr ((ii > jj ? ii - jj : jj - ii) <= Band)
					u_Func(i, j);
							});
						});
	}

	template<size_t Radius, typename Fn>
	constexpr void staticStencil(Fn&& u_Func) {
		constexpr int R = Radius;

		for (int y = -R; y <= R; ++y) {
			for (int x = -R; x <= R; ++x) {
				u_Func(std::integral_constant<int, x>{}, std::integral_constant<int, y>{});
			}
		}
	}

	template<size_t Begin, size_t End, typename T, typename Fn>
	constexpr T staticFold(T init, Fn&& u_Func) {
		T value = init;
		staticFor<Begin, End>([&](auto i) {
			value = u_Func(value, i); });
		return value;
	}

	template<size_t Begin, size_t End, typename T, typename Fn>
	constexpr void staticPrefixScan(T init, Fn&& u_Func) {
		T value = init;
		staticFor<Begin, End>([&](auto i) {
			value = u_Func(value, i); });
	}

	template<size_t Begin, size_t End, typename T, typename Fn>
	constexpr void staticSuffixScan(T init, Fn&& u_Func) {
		T value = init;
		staticForReverse<Begin, End>([&](auto i) {
			value = u_Func(value, i); });
	}

	template<size_t Begin, size_t End, typename T, typename Fn>
	constexpr T staticAccumulate(T init, Fn&& u_Func) {
		T value = init;
		staticFor<Begin, End>([&](auto i) {
			value += u_Func(i); });
		return value;
	}

	template<size_t Begin, size_t End, typename T, typename Transform, typename Reduce>
	constexpr T staticTransformReduce(T init, Transform&& transform, Reduce&& reduce) {
		T value = init;
		staticFor<Begin, End>([&](auto i) {
			value = reduce(value, transform(i)); });
		return value;
	}

	template<size_t Lanes, typename Fn>
	constexpr void staticLaneFor(Fn&& u_Func) {
		staticFor<0, Lanes>([&](auto lane) {
			u_Func(lane);
							});
	}

	template<size_t Begin, size_t End, size_t Width, typename Fn>
	constexpr void staticVectorBlocks(Fn&& u_Func) {
		staticStride<Begin, End, Width>([&](auto start) {u_Func(start); });
	}

	template<size_t Begin, size_t End, size_t Width, typename Fn>
	constexpr void staticVectorTile(Fn&& u_Func) {
		staticVectorBlocks<Begin, End, Width>([&](auto block) {u_Func(block); });
	}

	template<size_t Begin, size_t End, size_t Width, typename Fn>
	constexpr void staticVectorChunk(Fn&& u_Func) {
		staticStride<Begin, End, Width>([&](auto start) {
			constexpr size_t size = Width;
			u_Func(start, IndexT<size>{}); });
	}

	template<size_t Begin, size_t End, size_t Width, typename Fn>
	constexpr void staticTail(Fn&& u_Func) {
		constexpr size_t tailStart = Begin + ((End - Begin) / Width) * Width;
		staticFor<tailStart, End>([&](auto i) {
			u_Func(i); });
	}

	template<size_t Begin, size_t End, size_t Width, typename VectorFn, typename ScalarFn>
	constexpr void staticPackedIteration(VectorFn&& vectorFn, ScalarFn&& scalarFn) {
		staticVectorBlocks<Begin, End, Width>(
			[&](auto block) {
				vectorFn(block);
			});
		staticTail<Begin, End, Width>(
			[&](auto i) {
				scalarFn(i);
			});
	}

	template<size_t Stages, size_t Steps, typename Fn>
	constexpr void staticPipeline(Fn&& u_Func) {
		staticFor<0, Steps + Stages - 1>([&](auto cycle) {
			constexpr size_t c = cycle;
			staticFor<0, Stages>([&](auto stage) {
				constexpr size_t s = stage;
				constexpr size_t idx = c - s;
				if constexpr (c >= s && idx < Steps)
					u_Func(stage, IndexT<idx>{}); }); });
	}

	template<size_t Stage, size_t Steps, typename Fn>
	constexpr void staticStage(Fn&& u_Func) {
		staticFor<0, Steps>([&](auto step) {
			u_Func(IndexT<Stage>{}, step); });
	}

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticSchedule(Fn&& u_Func) {
		staticFor<Begin, End>([&](auto step) {
			u_Func(step); });
	}

	template<size_t Rows, size_t Cols, typename Fn>
	constexpr void staticWavefront(Fn&& u_Func) {
		staticFor<0, Rows + Cols - 1>([&](auto diag) {
			constexpr size_t d = diag;

			staticFor<0, Rows>([&](auto r) {
				constexpr size_t row = r;
				constexpr size_t col = d - row;

				if constexpr (col < Cols)
					u_Func(r, IndexT<col>{}); }); });
	}

	template<size_t N, typename Fn>
	constexpr void staticCascade(Fn&& u_Func) {
		staticFor<0, N>([&](auto step) {
			staticFor<0, step + 1>([&](auto inner) {
				u_Func(inner); }); });
	}

		template<size_t Count, typename FnA, typename FnB>
		constexpr void staticInterleave(FnA&& a, FnB&& b) {
			staticFor<0, Count>([&](auto i) {
				a(i);
				b(i); });
		}

	template<size_t Begin, size_t End, typename Fn>
	constexpr void staticDependencyChain(Fn&& u_Func) {
		staticFor<Begin, End>([&](auto i) {
			u_Func(i); });
	}
}
