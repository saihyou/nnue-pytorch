#ifndef _SFEN_STREAM_H_
#define _SFEN_STREAM_H_

#include "nnue_training_data_formats.h"
#include "../YaneuraOu/source/learn/learn.h"

#include <optional>
#include <fstream>
#include <string>
#include <memory>
#include <numeric>
//#include <ppl.h>

namespace training_data {

    using namespace binpack;

    static bool ends_with(const std::string& lhs, const std::string& end)
    {
        if (end.size() > lhs.size()) return false;

        return std::equal(end.rbegin(), end.rend(), lhs.rbegin());
    }

    static bool has_extension(const std::string& filename, const std::string& extension)
    {
        return ends_with(filename, "." + extension);
    }

    static std::string filename_with_extension(const std::string& filename, const std::string& ext)
    {
        if (ends_with(filename, ext))
        {
            return filename;
        }
        else
        {
            return filename + "." + ext;
        }
    }

    struct BasicSfenInputStream
    {
        virtual std::optional<TrainingDataEntry> next() = 0;
        virtual void fill(std::vector<TrainingDataEntry>& vec, std::size_t n)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                auto v = this->next();
                if (!v.has_value())
                {
                    break;
                }
                vec.emplace_back(*v);
            }
        }

        virtual bool eof() const = 0;
        virtual ~BasicSfenInputStream() {}
    };

    struct BinSfenInputStream : BasicSfenInputStream
    {
        static constexpr auto openmode = std::ios::in | std::ios::binary;
        static inline const std::string extension = "bin";

        BinSfenInputStream(std::string filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
            m_stream(filename, openmode),
            m_filename(filename),
            m_eof(!m_stream),
            m_cyclic(cyclic),
            m_skipPredicate(std::move(skipPredicate)),
	    m_engine(std::random_device()()),
            m_curr(0)
        {
	  m_stream.seekg(0, std::ios_base::end);
            const auto size = m_stream.tellg() / sizeof(Learner::PackedSfenValue);
            m_stream.seekg(0, std::ios_base::beg);
            m_data.resize(size);
            m_stream.read(reinterpret_cast<char*>(&m_data[0]), sizeof(Learner::PackedSfenValue) * size);
	    m_indexes.resize(size);
            std::iota(m_indexes.begin(), m_indexes.end(), 0);
            std::shuffle(m_indexes.begin(), m_indexes.end(), m_engine);
        }

        std::optional<TrainingDataEntry> next() override
        {
            Learner::PackedSfenValue e;
            bool reopenedFileOnce = false;
            for(;;)
            {
                if(m_stream.read(reinterpret_cast<char*>(&e), sizeof(Learner::PackedSfenValue)))
                {
                    auto entry = packedSfenValueToTrainingDataEntry(e);
                    if (!m_skipPredicate || !m_skipPredicate(entry))
                        return entry;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return std::nullopt;

                        m_stream = std::fstream(m_filename, openmode);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return std::nullopt;

                        continue;
                    }

                    m_eof = true;
                    return std::nullopt;
                }
            }
        }

        void fill(std::vector<TrainingDataEntry>& vec, std::size_t n) override
        {
            std::vector<Learner::PackedSfenValue> packedSfenValues(n);
            bool reopenedFileOnce = false;
            for (;;)
            {
                if (m_curr + n < m_indexes.size())
                {
                    vec.resize(n);
#if 0
                    concurrency::parallel_for(size_t(0), n, [&vec, &packedSfenValues](size_t i)
                        {
                            vec[i] = packedSfenValueToTrainingDataEntry(packedSfenValues[i]);
                        });
#else
		    for (size_t i = 0; i < n; i++)
                    {
                        vec[i] = packedSfenValueToTrainingDataEntry(m_data[m_indexes[m_curr++]]);
                    }
#endif
                    return;
                }
                else
                {
                    if (m_cyclic)
                    {
                        if (reopenedFileOnce)
                            return;

                        m_curr = 0;
                        std::shuffle(m_indexes.begin(), m_indexes.end(), m_engine);
                        reopenedFileOnce = true;
                        if (!m_stream)
                            return;

                        continue;
                    }

                    m_eof = true;
                    return;
                }
            }
        }

        bool eof() const override
        {
            return m_eof;
        }

        ~BinSfenInputStream() override {}

    private:
        std::fstream m_stream;
        std::string m_filename;
        bool m_eof;
        bool m_cyclic;
        std::function<bool(const TrainingDataEntry&)> m_skipPredicate;
      std::vector<Learner::PackedSfenValue> m_data;
        std::vector<unsigned int> m_indexes;
        std::mt19937_64 m_engine;
        unsigned int m_curr;
    };

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file(const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }

    inline std::unique_ptr<BasicSfenInputStream> open_sfen_input_file_parallel(int concurrency, const std::string& filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate = nullptr)
    {
        // TODO (low priority): optimize and parallelize .bin reading.
        if (has_extension(filename, BinSfenInputStream::extension))
            return std::make_unique<BinSfenInputStream>(filename, cyclic, std::move(skipPredicate));

        return nullptr;
    }
}

#endif
